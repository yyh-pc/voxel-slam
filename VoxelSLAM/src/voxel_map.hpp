#ifndef VOXEL_MAP2_HPP
#define VOXEL_MAP2_HPP

#include "tools.hpp"
#include "preintegration.hpp"
#include <thread>
#include <Eigen/Eigenvalues>
#include <unordered_set>
#include <mutex>

#include <ros/ros.h>
#include <fstream>

struct pointVar
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pnt;
  Eigen::Matrix3d var;
};

using PVec = vector<pointVar>;
using PVecPtr = shared_ptr<vector<pointVar>>;

void down_sampling_pvec(PVec &pvec, double voxel_size, pcl::PointCloud<PointType> &pl_keep)
{
  unordered_map<VOXEL_LOC, pair<pointVar, int>> feat_map;
  float loc_xyz[3];
  for(pointVar &pv: pvec)
  {
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pv.pnt[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter == feat_map.end())
    {
      feat_map[position] = make_pair(pv, 1);
    }
    else
    {
      pair<pointVar, int> &pp = iter->second;
      pp.first.pnt = (pp.first.pnt * pp.second + pv.pnt) / (pp.second + 1);
      pp.first.var = (pp.first.var * pp.second + pv.var) / (pp.second + 1);
      pp.second += 1;
    }
  }

  pcl::PointCloud<PointType>().swap(pl_keep);
  pl_keep.reserve(feat_map.size());
  PointType ap;
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
  {
    pointVar &pv = iter->second.first;
    ap.x = pv.pnt[0]; ap.y = pv.pnt[1]; ap.z = pv.pnt[2];
    ap.normal_x = pv.var(0, 0);
    ap.normal_y = pv.var(1, 1);
    ap.normal_z = pv.var(2, 2);
    pl_keep.push_back(ap);
  }

}

struct Plane
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> plane_var;
  float radius = 0;
  bool is_plane = false;

  Plane()
  {
    plane_var.setZero();
  }

};

Eigen::Vector4d min_point;
double min_eigen_value;
int max_layer = 2;
int max_points = 100;
double voxel_size = 1.0;
int min_ba_point = 20;
vector<double> plane_eigen_value_thre;

void Bf_var(const pointVar &pv, Eigen::Matrix<double, 9, 9> &bcov, const Eigen::Vector3d &vec)
{
  Eigen::Matrix<double, 6, 3> Bi;
  // Eigen::Vector3d &vec = pv.world;
  Bi << 2*vec(0),        0,        0,
          vec(1),   vec(0),        0,
          vec(2),        0,   vec(0),
               0, 2*vec(1),        0,
               0,   vec(2),   vec(1),
               0,        0, 2*vec(2);
  Eigen::Matrix<double, 6, 3> Biup = Bi * pv.var;
  bcov.block<6, 6>(0, 0) = Biup * Bi.transpose();
  bcov.block<6, 3>(0, 6) = Biup;
  bcov.block<3, 6>(6, 0) = Biup.transpose();
  bcov.block<3, 3>(6, 6) = pv.var;
}

// The LiDAR BA factor in optimization
// note:用于存储 LiDAR 点云聚类信息（voxel/cluster）和对应的误差项
class LidarFactor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<PointCluster> sig_vecs;
  vector<vector<PointCluster>> plvec_voxels;
  vector<double> coeffs;
  PLV(3) eig_values; PLM(3) eig_vectors;
  vector<PointCluster> pcr_adds;
  int win_size;

  LidarFactor(int _w): win_size(_w){}

  void push_voxel(vector<PointCluster> &vec_orig, PointCluster &fix, double coe, Eigen::Vector3d &eig_value, Eigen::Matrix3d &eig_vector, PointCluster &pcr_add)
  {
    plvec_voxels.push_back(vec_orig);
    sig_vecs.push_back(fix);
    coeffs.push_back(coe);
    eig_values.push_back(eig_value);
    eig_vectors.push_back(eig_vector);
    pcr_adds.push_back(pcr_add);
  }

  void acc_evaluate2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<PointCluster> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a=head; a<end; a++)
    {
      vector<PointCluster> &sig_orig = plvec_voxels[a];
      double coe = coeffs[a];

      // PointCluster sig = sig_vecs[a];
      // for(int i=0; i<win_size; i++)
      // if(sig_orig[i].N != 0)
      // {
      //   sig_tran[i].transform(sig_orig[i], xs[i]);
      //   sig += sig_tran[i];
      // }

      // const Eigen::Vector3d &vBar = sig.v / sig.N;
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      // const Eigen::Vector3d &lmbd = saes.eigenvalues();
      // const Eigen::Matrix3d &U = saes.eigenvectors();
      // int NN = sig.N;

      Eigen::Vector3d lmbd = eig_values[a];
      Eigen::Matrix3d U = eig_vectors[a];
      int NN = pcr_adds[a].N;
      Eigen::Vector3d vBar = pcr_adds[a].v / NN;

      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};
      Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i=0; i<3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i=0; i<win_size; i++)
      // for(int i=1; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        Eigen::Matrix3d Pi = sig_orig[i].P;
        Eigen::Vector3d vi = sig_orig[i].v;
        Eigen::Matrix3d Ri = xs[i].R;
        double ni = sig_orig[i].N;

        Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
        Eigen::Vector3d RiTuk = Ri.transpose() * uk;
        Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

        Eigen::Vector3d PiRiTuk = Pi * RiTuk;
        viRiTuk[i] = vihat * RiTuk;
        viRiTukukT[i] = viRiTuk[i] * uk.transpose();

        Eigen::Vector3d ti_v = xs[i].p - vBar;
        double ukTti_v = uk.dot(ti_v);

        Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
        Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
        Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
        Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
        Auk[i] /= NN;

        const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
        JacT.block<6, 1>(6*i, 0) += coe * jjt;

        const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
        Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
        Hb.block<3, 3>(0, 0) += 2.0/NN * (combo1 - RiTukhat*Pi) * RiTukhat - 2.0/NN/NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

        Hess.block<6, 6>(6*i, 6*i) += coe * Hb;
      }

      for(int i=0; i<win_size-1; i++)
      // for(int i=1; i<win_size-1; i++)
      if(sig_orig[i].N != 0)
      {
        double ni = sig_orig[i].N;
        for(int j=i+1; j<win_size; j++)
        if(sig_orig[j].N != 0)
        {
          double nj = sig_orig[j].N;
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
          Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
          Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
          Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
          Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

          Hess.block<6, 6>(6*i, 6*j) += coe * Hb;
        }
      }

      residual += coe * lmbd[kk];
    }

    for(int i=1; i<win_size; i++)
      for(int j=0; j<i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();

  }

  void evaluate_only_residual(const vector<IMUST> &xs, int head, int end, double &residual)
  {
    residual = 0;
    // vector<PointCluster> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    // int gps_size = plvec_voxels.size();
    PointCluster pcr;

    for(int a=head; a<end; a++)
    {
      const vector<PointCluster> &sig_orig = plvec_voxels[a];
      PointCluster sig = sig_vecs[a];

      for(int i=0; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        pcr.transform(sig_orig[i], xs[i]);
        sig += pcr;
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      // Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P - sig.v * vBar.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();

      // centers[a] = vBar;
      eig_values[a] = saes.eigenvalues();
      eig_vectors[a] = saes.eigenvectors();
      pcr_adds[a] = sig;
      // Ns[a] = sig.N;

      residual += coeffs[a] * lmbd[kk];
    }

  }

  void clear()
  {
    sig_vecs.clear(); plvec_voxels.clear();
    eig_values.clear(); eig_vectors.clear();
    pcr_adds.clear(); coeffs.clear();
  }

  ~LidarFactor(){}

};

/**
 * @brief The LM optimizer for LiDAR BA
 * note:用于全局建图模块
 * 对 LidarFactor 中的 LiDAR 点云信息进行 BA 优化。
   使用 Levenberg-Marquardt (LM) 方法求解 pose 更新。
   提供多线程加速。
 *
 */
class Lidar_BA_Optimizer
{
public:
  int win_size, jac_leng, thd_num = 2;

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    // int thd_num = 4;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    // for(int i=0; i<tthd_num; i++)
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for(int i=0; i<tthd_num; i++)
    {
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    double residual1 = 0;
    // voxhess.evaluate_only_residual(x_stats, 0, voxhess.plvec_voxels.size(), residual1);

    // int thd_num = 2;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      printf("Too Less Voxel"); exit(0);
    }
    vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0)
        mthreads[i]->join();
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual1 += residuals[i];
      delete mthreads[i];
    }

    return residual1;
  }

  bool damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd* hess, vector<double> &resis, int max_iter = 3, bool is_display = false)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);
    hess->resize(jac_leng, jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    bool is_converge = true;

    // double tt1 = ros::Time::now().toSec();
    // for(int i=0; i<10; i++)
    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        residual1 = divide_thread(x_stats, voxhess, Hess, JacT);
        *hess = Hess;
      }

      if(i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(6*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(6*j+3, 0);
      }
      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);

      residual2 = only_residual(x_stats_temp, voxhess);

      q = (residual1-residual2);
      if(is_display)
        printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
        is_converge = false;
      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }
    resis.push_back(residual2);
    return is_converge;
  }

};

double imu_coef = 1e-4;
// double imu_coef = 1e-8;
#define DVEL 6

/**
 * @brief The LiDAR-Inertial BA optimizer
 * note: 用于局部BA
 * 将 LiDAR BA 与 IMU 预积分约束结合，做 LiDAR-Inertial BA
 * 在优化中加入 IMU 残差项，并使用 imu_coef 缩放权重
 */
class LI_BA_Optimizer
{
public:
  int win_size, jac_leng, imu_leng;

  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for(int i=0; i<win_size; i++)
    {
      JacT.block<DVEL, 1>(i*DIM, 0) += js.block<DVEL, 1>(i*DVEL, 0);
      for(int j=0; j<win_size; j++)
        Hess.block<DVEL, DVEL>(i*DIM, j*DIM) += hs.block<DVEL, DVEL>(i*DVEL, j*DVEL);
    }
  }

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);
    vector<double> resis(thd_num, 0);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    vector<thread*> mthreads(tthd_num);
    // for(int i=0; i<tthd_num; i++)
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part * (i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    for(int i=0; i<win_size-1; i++)
    {
      jtj.setZero(); gg.setZero();
      residual += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, true);
      Hess.block<DIM*2, DIM*2>(i*DIM, i*DIM) += jtj;
      JacT.block<DIM*2, 1>(i*DIM, 0) += gg;
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity(); rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual *= (imu_coef * 0.5);

    // printf("resi: %lf\n", residual);

    for(int i=0; i<tthd_num; i++)
    {
      // mthreads[i]->join();
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor)
  {
    double residual1 = 0, residual2 = 0;
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    int thd_num = 5;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      // printf("Too Less Voxel"); exit(0);
      thd_num = 1;
    }
    vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<win_size-1; i++)
      residual1 += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i+1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0)
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual2 += residuals[i];
    }

    return (residual1 + residual2);
  }

  /**
   * @brief 带阻尼的滑动窗口非线性最小二乘优化,对滑动窗口的位姿、速度和 IMU 偏差做 联合优化，融合激光和 IMU 信息
   *
   * @param x_stats  滑动窗口状态, 通过引用传入,可在函数中直接更新
   * @param voxhess  包含滑动窗口点云的 Lidar 信息矩阵和 Hessian，用于计算激光残差
   * @param imus_factor  滑动窗口内每帧 IMU 预积分对象，用于 IMU 残差和状态更新
   * @param hess      输出 Hessian 矩阵指针，用于返回当前窗口的完整 Hessian
   */
  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd* hess)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;
    imu_leng = win_size * DIM;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng);
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);
    hess->resize(imu_leng, imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    double hesstime = 0;
    double resitime = 0;

    // for(int i=0; i<10; i++)
    for(int i=0; i<3; i++)
    {
      if(is_calc_hess)
      {
        double tm = ros::Time::now().toSec();
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        hesstime += ros::Time::now().toSec() - tm;
        *hess = Hess;
      }

      Hess.topRows(DIM).setZero();
      Hess.leftCols(DIM).setZero();
      Hess.block<DIM, DIM>(0, 0).setIdentity();
      JacT.head(DIM).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM*j+3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM*j+6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM*j+9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM*j+12, 0);
      }

      for(int j=0; j<win_size-1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM*j, 0));

      double q1 = 0.5 * dxi.dot(u*D*dxi-JacT);

      double tl1 = ros::Time::now().toSec();
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      double tl2 = ros::Time::now().toSec();
      // printf("onlyresi: %lf\n", tl2-tl1);
      resitime += tl2 - tl1;

      q = (residual1-residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for(int j=0; j<win_size-1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }

      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }

    // printf("ba: %lf %lf %zu\n", hesstime, resitime, voxhess.plvec_voxels.size());

  }

};

/**
 * @brief The LiDAR-Inertial BA optimizer with gravity optimization
 * note:用于初始化
 * 在 LI_BA_Optimizer 基础上增加了重力向量优化
 */
class LI_BA_OptimizerGravity
{
public:
  int win_size, jac_leng, imu_leng;

  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for(int i=0; i<win_size; i++)
    {
      JacT.block<DVEL, 1>(i*DIM, 0) += js.block<DVEL, 1>(i*DVEL, 0);
      for(int j=0; j<win_size; j++)
        Hess.block<DVEL, DVEL>(i*DIM, j*DIM) += hs.block<DVEL, DVEL>(i*DVEL, j*DVEL);
    }
  }

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);
    vector<double> resis(thd_num, 0);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    vector<thread*> mthreads(tthd_num);
    // for(int i=0; i<tthd_num; i++)
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part * (i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    Eigen::MatrixXd jtj(2*DIM+3, 2*DIM+3);
    Eigen::VectorXd gg(2*DIM+3);

    for(int i=0; i<win_size-1; i++)
    {
      jtj.setZero(); gg.setZero();
      residual += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i+1], jtj, gg, true);
      Hess.block<DIM*2, DIM*2>(i*DIM, i*DIM) += jtj.block<2*DIM, 2*DIM>(0, 0);
      Hess.block<DIM*2, 3>(i*DIM, imu_leng-3) += jtj.block<2*DIM, 3>(0, 2*DIM);
      Hess.block<3, DIM*2>(imu_leng-3, i*DIM) += jtj.block<3, 2*DIM>(2*DIM,0);
      Hess.block<3, 3>(imu_leng-3, imu_leng-3) += jtj.block<3, 3>(2*DIM, 2*DIM);

      JacT.block<DIM*2, 1>(i*DIM, 0) += gg.head(2*DIM);
      JacT.tail(3) += gg.tail(3);
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity(); rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual *= (imu_coef * 0.5);

    // printf("resi: %lf\n", residual);

    for(int i=0; i<tthd_num; i++)
    {
      // mthreads[i]->join();
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor)
  {
    double residual1 = 0, residual2 = 0;
    Eigen::MatrixXd jtj(2*DIM, 2*DIM);
    Eigen::VectorXd gg(2*DIM);

    int thd_num = 5;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      // printf("Too Less Voxel"); exit(0);
      thd_num = 1;
    }
    vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<win_size-1; i++)
      residual1 += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i+1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0)
      {
        mthreads[i]->join(); delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual2 += residuals[i];
    }

    return (residual1 + residual2);
  }

  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE*> &imus_factor, vector<double> &resis, Eigen::MatrixXd* hess, int max_iter = 2)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;
    imu_leng = win_size * DIM + 3;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng);
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        *hess = Hess;
      }

      if(i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();

      // Hess.rightCols(3).setZero();
      // Hess.bottomRows(3).setZero();
      // Hess.block<3, 3>(imu_leng-3, imu_leng-3).setIdentity();
      // JacT.tail(3).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      x_stats_temp[0].g += dxi.tail(3);
      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM*j+3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM*j+6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM*j+9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM*j+12, 0);
        x_stats_temp[j].g = x_stats_temp[0].g;
      }

      for(int j=0; j<win_size-1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM*j, 0));

      double q1 = 0.5 * dxi.dot(u*D*dxi-JacT);
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      q = (residual1-residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for(int j=0; j<win_size-1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }
      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;

    }
    resis.push_back(residual2);

  }

};

// 10 scans merge into a keyframe
struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IMUST x0;
  pcl::PointCloud<PointType>::Ptr plptr;
  int exist;
  int id, mp;
  float jour;

  Keyframe(IMUST &_x0): x0(_x0), exist(0)
  {
    plptr.reset(new pcl::PointCloud<PointType>());
  }

  void generate(pcl::PointCloud<PointType> &pl_send, Eigen::Matrix3d rot = Eigen::Matrix3d::Identity(), Eigen::Vector3d tra = Eigen::Vector3d(0, 0, 0))
  {
    Eigen::Vector3d v3;
    for(PointType ap: plptr->points)
    {
      v3 << ap.x, ap.y, ap.z;
      v3 = rot * v3 + tra;
      ap.x = v3[0]; ap.y = v3[1]; ap.z = v3[2];
      pl_send.push_back(ap);
    }
  }

};

/**
 * @brief The sldingwindow in each voxel nodes
 * SlideWindow 是某一个体素节点内部的滑动窗口缓存
 * 用于暂存「最近 N 帧」在该体素中的观测数据
 * 只在该体素参与最近 wdsize 帧优化 / 统计时才有意义，当体素被冻结 / 边缘化 / 降级为历史地图后可能会被回收
 * 注意OctoTree是长期的，但是SlideWindow 的生命周期可能比较短
 */
class SlideWindow
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<PVec> points;               // 每一帧的原始点（带协方差、局部坐标）
  vector<PointCluster> pcrs_local;   // ?:这帧的统计信息（平均点、协方差），方便后续快速建模或优化

  SlideWindow(int wdsize)
  {
    pcrs_local.resize(wdsize);
    points.resize(wdsize);
    for(int i=0; i<wdsize; i++)
      points[i].reserve(20);
  }

  void resize(int wdsize)
  {
    if(points.size() != wdsize)
    {
      points.resize(wdsize);
      pcrs_local.resize(wdsize);
    }
  }

  void clear()
  {
    int wdsize = points.size();
    for(int i=0; i<wdsize; i++)
    {
      points[i].clear();
      pcrs_local[i].clear();
    }
  }

};

// The octotree map for odometry and local mapping
// You can re-write it in your own project
int* mp;
class OctoTree
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  //note:第一类数据：滑动窗口：当前体素最近 N 帧的观测（IMU系下）
  SlideWindow* sw = nullptr;   //八叉树子体素划分(recut中的subdivide)、构建局部BA优化的voxhess因子(recut中的tras_opt)、用于边缘化margi

  //note:第二类数据：当前用于几何判断 / 优化构建的候选点统计量
  PointCluster pcr_add;    // (point_fix + 当前滑窗点)的统计量，用于平面判断和更新、构建局部BA优化的voxhess因子[recut、margi]

  Eigen::Matrix<double, 9, 9> cov_add;   //用于平面更新plane_update

  //note:第三类数据：固定点簇和固定点(世界系)
  PointCluster pcr_fix;   //体素长期的固定点簇，主要来自历史关键帧的cut以及滑窗的边缘化，点簇的点数上限max_points(100)，用于构建局部BA优化的voxhess因子、用于边缘化margi
  PVec point_fix;         //逐点级别的历史点缓存，用于体素的八叉树划分fix_divide，构建子体素

  int layer, octo_state, wdsize;
  OctoTree* leaves[8];                  //八叉树子节点
  double voxel_center[3];               //当前体素中心坐标
  double jour = 0;                      //观测到当前体素时的累积里程
  float quater_length;

  bool isexist = false;                 //?:这个标志位是什么意思

  //平面特征和特征判定相关
  Plane plane;
  Eigen::Vector3d eig_value;
  Eigen::Matrix3d eig_vector;

  int last_num = 0, opt_state = -1;
  mutex mVox;

  OctoTree(int _l, int _w) : layer(_l), wdsize(_w), octo_state(0)
  {
    for(int i=0; i<8; i++) leaves[i] = nullptr;
    cov_add.setZero();

    // ins = 255.0*rand()/(RAND_MAX + 1.0f);
  }

  inline void push(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow*> &sws)
  {
    mVox.lock();
    // 每个 OctoTree 叶节点只有一个 SlideWindow，获取 / 创建 SlideWindow
    if(sw == nullptr)
    {
      //从sws里获取一个
      if(sws.size() != 0)
      {
        sw = sws.back();
        sws.pop_back();
        sw->resize(wdsize);   //滑窗大小为10
      }
      else   //没有就重新new一个
        sw = new SlideWindow(wdsize);
    }

    // 标记当前体素有效
    if(!isexist) isexist = true;

    // note:初始化在main函数
    int mord = mp[ord];
    // std::cout << "mord: " << mord << ", mp[ord]: " << mp[ord] << std::endl;

    //滑窗存IMU系点
    if(layer < max_layer)
      sw->points[mord].push_back(pv);

    //滑窗存点簇(统计量信息)
    sw->pcrs_local[mord].push(pv.pnt);

    // 八叉树存储世界系点
    pcr_add.push(pw);

    // 输入IMU系下点的协方差、世界坐标，计算该点对voxel不确定性，累加
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pw);
    cov_add += Bi;
    mVox.unlock();
  }

  inline void push_fix(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pv.pnt);
    cov_add += Bi;
  }

  inline void push_fix_novar(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
  }

  inline bool plane_judge(Eigen::Vector3d &eig_values)
  {
    // return (eig_values[0] < min_eigen_value);
    return (eig_values[0] < min_eigen_value && (eig_values[0]/eig_values[2])<plane_eigen_value_thre[layer]);
  }

  /**
   * @brief
   *
   * @param ord  当前点来自滑窗中的 第几帧
   * @param pv   点在IMU系 + 协方差
   * @param pw   同一个点在世界坐标系
   * @param sws  SlideWindow 对象池
   */
  void allocate(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow*> &sws)
  {
    if(octo_state == 0)
    {
      push(ord, pv, pw, sws);
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate(ord, pv, pw, sws);
    }

  }

  void allocate_fix(pointVar &pv)
  {
    if(octo_state == 0)
    {
      push_fix_novar(pv);
    }
    else if(layer < max_layer)
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate_fix(pv);
    }
  }

  /**
   * @brief 将固定地图点切分到子体素
   * note:point_fix来源于历史关键帧的cut，以及边缘化扫描帧
   * @param sws
   */
  void fix_divide(vector<SlideWindow*> &sws)
  {
    // 遍历当前体素内的所有固定点
    for(pointVar &pv: point_fix)
    {
      //将三轴判断映射为子体素索引（0–7）
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      //如果子体素不存在，则创建(由此可见八叉树是按需生长的)
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);  //子体素层级+1，最大为2
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      //note:子体素加入固定点,注意这里用的是push_fix
      leaves[leafnum]->push_fix(pv);
    }

  }

  //把当前体素内所有滑窗帧的点，按世界坐标重新分发到子体素
  void subdivide(int si, IMUST &xx, vector<SlideWindow*> &sws)
  {
    for(pointVar &pv: sw->points[mp[si]])
    {
      //滑窗内的点从IMU系转到世界坐标系
      Eigen::Vector3d pw = xx.R * pv.pnt + xx.p;
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      //创建子体素
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      //note:滑窗内的点cut进子体素,注意这里用的是push
      leaves[leafnum]->push(si, pv, pw, sws);
    }
  }

  void plane_update()
  {
    plane.center = pcr_add.v / pcr_add.N;
    int l = 0;
    Eigen::Vector3d u[3] = {eig_vector.col(0), eig_vector.col(1), eig_vector.col(2)};
    double nv = 1.0 / pcr_add.N;

    Eigen::Matrix<double, 3, 9> u_c; u_c.setZero();
    for(int k=0; k<3; k++)
    if(k != l)
    {
      Eigen::Matrix3d ukl = u[k] * u[l].transpose();
      Eigen::Matrix<double, 1, 9> fkl;
      fkl.head(6) << ukl(0, 0), ukl(1, 0)+ukl(0, 1), ukl(2, 0)+ukl(0, 2),
                     ukl(1, 1), ukl(1, 2)+ukl(2, 1),           ukl(2, 2);
      fkl.tail(3) = -(u[k].dot(plane.center) * u[l] + u[l].dot(plane.center) * u[k]);

      u_c += nv / (eig_value[l]-eig_value[k]) * u[k] * fkl;
    }

    Eigen::Matrix<double, 3, 9> Jc = u_c * cov_add;
    plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
    Eigen::Matrix3d Jc_N = nv * Jc.block<3, 3>(0, 6);
    plane.plane_var.block<3, 3>(0, 3) = Jc_N;
    plane.plane_var.block<3, 3>(3, 0) = Jc_N.transpose();
    plane.plane_var.block<3, 3>(3, 3) = nv * nv * cov_add.block<3, 3>(6, 6);
    plane.normal = u[0];
    plane.radius = eig_value[2];
  }

  /**
   * @brief 对当前体素节点，自适应体素分辨率 + 平面提取
   *
   * @param win_count
   * @param x_buf
   * @param sws
   * 对当前 OctoTree 节点累计的 pcr_add 点云统计量进行平面可拟合性判断，如可以则recut结束
   * 如不能拟合平面则将固定点(来自关键帧和边缘化扫描帧)和滑窗点重新切分进子体素，去递归拟合平面
   */
  void recut(int win_count, vector<IMUST> &x_buf, vector<SlideWindow*> &sws)
  {
    //step: 0代表该体素还未处理，这一步是判断体素是否需要八叉树划分
    if(octo_state == 0)
    {
      //?:初始化的时候layer都是0，这个判定是什么作用?
      if(layer >= 0)
      {
        //opt_state = -1：初始化为尚未形成可用优化约束的状态
        opt_state = -1;

        // 当前体素内的点数少于5个，不足以支持PCA
        if(pcr_add.N <= min_point[layer])
        {
          plane.is_plane = false;
          return;
        }

        // 体素不存在 / 没有滑窗 → 返回
        if(!isexist || sw == nullptr) return;

        // note:对体素内点（世界系下）做 PCA, 求解plane
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
        eig_value  = saes.eigenvalues();
        eig_vector = saes.eigenvectors();
        plane.is_plane = plane_judge(eig_value);

        // 若为平面则直接返回不再细分
        if(plane.is_plane)
        {
          return;
        }
        //若达到最大层级也不再细分，layer = 1可以切分为8个，layer = 2可以切分为8*8个
        else if(layer >= max_layer)
          return;
      }

      //note:执行到这里说明父节点体素无法拟合出有效平面，需要进行体素自适应划分

      //step:若父体素存在固定点，则将固定点切分至子体素，父体素清空固定点
      if(pcr_fix.N != 0)
      {
        fix_divide(sws);
        PVec().swap(point_fix);
      }

      //step:父节点体素滑动窗口内的点切分至子体素
      for(int i=0; i<win_count; i++)
        subdivide(i, x_buf[i], sws);

      //父节点体素滑窗清空回收至sws滑动窗口池
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;

      //标记该体素已经进行了体素划分
      octo_state = 1;
    }

    // step:对子体素进行递归recut平面拟合
    for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(win_count, x_buf, sws);

  }

  /**
   * @brief 将被边缘化帧的点云信息，压缩进一个长期稳定的统计表示（平面 + 累计点簇）
   *
   * note:如何理解这个边缘化在干什么
   * 在滑窗前移、状态即将被边缘化之前，
    将边缘化帧在各体素内的局部点簇统一变换到世界坐标系，
    用于更新或继承体素内已有的平面统计模型；
    随后，根据体素的稳定程度决定信息的保留方式：
    若体素尚未达到稳定状态，则将边缘化帧的点簇吸收到体素的固定点簇中，使其作为长期地图信息持续参与后续建模与约束；
    若体素已趋于稳定并达到固定点数量上限，则不再吸收新的固定点，其对应的统计贡献将从当前有效统计中移除；
    最终，清除边缘化帧在滑窗内维护的局部点簇与原始点信息，
    以完成该帧在体素层面的信息压缩与边缘化处理。
   *
   * @param win_count 当前滑窗帧数
   * @param mgsize    本次被边缘化的帧数
   * @param x_buf     滑动窗口位姿
   * @param vox_opt   滑窗的lidarBA优化因子
   */
  void margi(int win_count, int mgsize, vector<IMUST> &x_buf, const LidarFactor &vox_opt)
  {
    //当前为叶子节点
    if(octo_state == 0 && layer>=0)
    {
      //体素是否有效
      if(!isexist || sw == nullptr) return;

      mVox.lock();
      vector<PointCluster> pcrs_world(wdsize);   //准备临时世界坐标点簇缓存

      //note:索引的合法性检查，opt_state 是参与此次局部BA的体素在voxhess中的索引
      if(opt_state >= int(vox_opt.pcr_adds.size()))
      {
        printf("Error: opt_state: %d %zu\n", opt_state, vox_opt.pcr_adds.size());
        exit(0);
      }

      // 对于参与了此次局部BA的体素，可以直接复用voxhess中的结果
      if(opt_state >= 0)
      {
        pcr_add = vox_opt.pcr_adds[opt_state];
        eig_value  = vox_opt.eig_values[opt_state];
        eig_vector = vox_opt.eig_vectors[opt_state];
        opt_state = -1;         //使用后立即置位为-1

        // 将边缘化帧的统计量转化到世界坐标系下, mgsize为1
        for(int i=0; i<mgsize; i++)
        {
          if(sw->pcrs_local[mp[i]].N != 0)
          {
            pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
          }
        }

      }
      else  //对于未参与此次BA优化的体素来说(即opt_state==-1)
      {
        // pcr_fix为历史关键帧、边缘化点形成的固定点簇
        pcr_add = pcr_fix;

        // 加上转到世界系下的边缘帧的点簇
        for(int i=0; i<win_count; i++)
        {
          if(sw->pcrs_local[mp[i]].N != 0)
          {
            pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
            pcr_add += pcrs_world[i];
          }
        }

        // note:opt_state为-1的分支也是可能存在平面的只是没有达到参与此次BA优化的阈值
        // note:margi的核心目标是为了维护地图中已有平面eig_value、eig_vector
        if(plane.is_plane)
        {
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
          eig_value = saes.eigenvalues();
          eig_vector = saes.eigenvectors();
        }

      }

      //平面模型的增量式更新策略(当固定点簇未满100且当前体素新增超过5个点或者上次体素内点小于10)
      if(pcr_fix.N < max_points && plane.is_plane)
      {
        if(pcr_add.N - last_num >= 5 || last_num <= 10)
        {
            plane_update();          //利用pcr_add进行平面更新
            last_num = pcr_add.N;
        }
      }


      // 将边缘化帧的点并入固定点
      if(pcr_fix.N < max_points)    //固定点簇的点数还未达到100
      {
        //将边缘化帧的世界系点簇加入pcr_fix，将边缘化帧的滑窗点转到世界系加入point_fix
        for(int i=0; i<mgsize; i++)
        {
          if(pcrs_world[i].N != 0)
          {
            pcr_fix += pcrs_world[i];
            for(pointVar pv: sw->points[mp[i]])
            {
                pv.pnt = x_buf[i].R * pv.pnt + x_buf[i].p;
                point_fix.push_back(pv);
            }
          }

        }

      }
      else   //note:当固定点簇的点数大于100时，体素被认为统计量已收敛,因此不会再固化新点、也不会保存边缘化帧的统计信息
      {
        for(int i=0; i<mgsize; i++)
          if(pcrs_world[i].N != 0)
            pcr_add -= pcrs_world[i];

        if(point_fix.size() != 0)
          PVec().swap(point_fix);
      }

      //note:清除当前体素滑窗数据的边缘帧
      for(int i=0; i<mgsize; i++)
      if(sw->pcrs_local[mp[i]].N != 0)
      {
        sw->pcrs_local[mp[i]].clear();
        sw->points[mp[i]].clear();
      }

      // ?:当固定点簇的点数大于全部点数时isexist标志位置为false，边缘化完成之后会对该体素进行回收
      if(pcr_fix.N >= pcr_add.N)
        isexist = false;
      else
        isexist = true;

      mVox.unlock();
    }
    else   //非叶子体素：递归处理子节点
    {
      isexist = false;
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->margi(win_count, mgsize, x_buf, vox_opt);
        isexist = isexist || leaves[i]->isexist;
      }
    }

  }

  /**
   * @brief Extract the LiDAR factor
   * 在体素 OctoTree 中，自顶向下遍历所有叶子体素，将满足平面退化约束的体素转换为LiDAR BA优化因子，并累积进 LidarFactor（Hessian / 残差系统）中
   * @param vox_opt
   */
  void tras_opt(LidarFactor &vox_opt)
  {
    // 情况 1：当前节点是叶子体素
    if(octo_state == 0)
    {
      // 基本有效性检查
        // layer >= 0
        // isexist
        // plane.is_plane    : 当前体素成功拟合出平面
        // sw != nullptr     : 已绑定滑窗
      if(layer >= 0 && isexist && plane.is_plane && sw!=nullptr)
      {
        // 利用平面 PCA 特征值比例判断退化, eig_value[0]/eig_value[1] 过大 → 平面法向不稳定 → 跳过该体素
        if(eig_value[0]/eig_value[1] > 0.12)
          return;

        double coe = 1;                           // 当前体素对应的优化权重（目前固定为 1）
        vector<PointCluster> pcrs(wdsize);        // 为滑窗内每一帧准备一个 PointCluster
        for(int i=0; i<wdsize; i++)
          pcrs[i] = sw->pcrs_local[mp[i]];        // 从 SlideWindow 中取出对应的点簇
        opt_state = vox_opt.plvec_voxels.size();  // 记录当前 voxel 在LidarFactor中的索引

        /**
         * @brief 构建局部BA优化的lidar因子
         * pcrs：滑动窗口的点簇(IMU系)
         * pcr_fix: 体素内的固定点簇，主要来自历史关键帧的cut以及滑窗的边缘化
         * coe: 当前体素的初始权重
         * eig_value、eig_vector：当前体素PCA得到的特征值和特征向量
         * pcr_add：当前用于体素平面判定的点簇
         */
        vox_opt.push_voxel(pcrs, pcr_fix, coe, eig_value, eig_vector, pcr_add);
      }

    }
    else   // 情况 2：非叶子节点，递归下探
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
    }


  }

  int match(Eigen::Vector3d &wld, Plane* &pla, double &max_prob, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc)
  {
    int flag = 0;
    if(octo_state == 0)
    {
      if(plane.is_plane)
      {
        float dis_to_plane = fabs(plane.normal.dot(wld - plane.center));
        float dis_to_center = (plane.center - wld).squaredNorm();
        float range_dis = (dis_to_center - dis_to_plane * dis_to_plane);
        if(range_dis <= 3*3*plane.radius)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = wld - plane.center;
          J_nq.block<1, 3>(0, 3) = -plane.normal;
          double sigma_l = J_nq * plane.plane_var * J_nq.transpose();
          sigma_l += plane.normal.transpose() * var_wld * plane.normal;
          if(dis_to_plane < 3 * sqrt(sigma_l))
          {
            // float prob = 1 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
            // if(prob > max_prob)
            {
              oc = this;
              sigma_d = sigma_l;
              // max_prob = prob;
              pla = &plane;
            }

            flag = 1;
          }
        }
      }
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(wld[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      // for(int i=0; i<8; i++)
      // if(leaves[i] != nullptr)
      // {
      //   int flg = leaves[i]->match(wld, pla, max_prob, var_wld);
      //   if(i == leafnum)
      //     flag = flg;
      // }

      if(leaves[leafnum] != nullptr)
        flag = leaves[leafnum]->match(wld, pla, max_prob, var_wld, sigma_d, oc);

      // for(int i=0; i<8; i++)
      //   if(leaves[i] != nullptr)
      //     leaves[i]->match(pv, pla, max_prob, var_wld);
    }

    return flag;
  }

  void tras_ptr(vector<OctoTree*> &octos_release)
  {
    if(octo_state == 1)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        octos_release.push_back(leaves[i]);
        leaves[i]->tras_ptr(octos_release);
      }
    }
  }

  // ~OctoTree()
  // {
  //   for(int i=0; i<8; i++)
  //   if(leaves[i] != nullptr)
  //   {
  //     delete leaves[i];
  //     leaves[i] = nullptr;
  //   }
  // }

  // Extract the point cloud map for debug
  void tras_display(int win_count, pcl::PointCloud<PointType> &pl_fixd, pcl::PointCloud<PointType> &pl_wind, vector<IMUST> &x_buf)
  {
    if(octo_state == 0)
    {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
      Eigen::Matrix3d eig_vectors = saes.eigenvectors();
      Eigen::Vector3d eig_values  = saes.eigenvalues();

      PointType ap;
      // ap.intensity = ins;

      if(plane.is_plane)
      {
        // if(pcr_add.N-pcr_fix.N < min_ba_point) return;
        // if(eig_value[0]/eig_value[1] > 0.1)
        //   return;

        // for(pointVar &pv: point_fix)
        // {
        //   Eigen::Vector3d pvec = pv.pnt;
        //   ap.x = pvec[0];
        //   ap.y = pvec[1];
        //   ap.z = pvec[2];
        //   ap.normal_x = sqrt(eig_values[0]);
        //   ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
        //   ap.normal_z = pcr_add.N;
        //   ap.curvature = pcr_add.N - pcr_fix.N;
        //   pl_fixd.push_back(ap);
        // }

        for(int i=0; i<win_count; i++)
        for(pointVar &pv: sw->points[mp[i]])
        {
          Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
          ap.x = pvec[0]; ap.y = pvec[1]; ap.z = pvec[2];
          // ap.normal_x = sqrt(eig_values[0]);
          // ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
          // ap.normal_z = pcr_add.N;
          // ap.curvature = pcr_add.N - pcr_fix.N;
          pl_wind.push_back(ap);
        }
      }

    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(win_count, pl_fixd, pl_wind, x_buf);
    }

  }

  bool inside(Eigen::Vector3d &wld)
  {
    double hl = quater_length * 2;
    return (wld[0] >= voxel_center[0] - hl &&
            wld[0] <= voxel_center[0] + hl &&
            wld[1] >= voxel_center[1] - hl &&
            wld[1] <= voxel_center[1] + hl &&
            wld[2] >= voxel_center[2] - hl &&
            wld[2] <= voxel_center[2] + hl);
  }

  void clear_slwd(vector<SlideWindow*> &sws)
  {
    if(octo_state != 0)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->clear_slwd(sws);
      }
    }

    if(sw != nullptr)
    {
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
    }

  }

};

/**
 * @brief 点 → 体素 → Octree → 滑窗
 *
 */
void cut_voxel(
  unordered_map<VOXEL_LOC, OctoTree*> &feat_map,       //全局体素地图
  PVecPtr pvec,                                        //去畸变点云(IMU系)
  int win_count,                                       //滑窗索引计数
  unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map,   //当前滑窗用到的体素子集
  int wdsize,                                          //滑窗大小，用于 OctoTree / SlideWindow 初始化
  PLV(3) &pwld,                                        //当前帧世界系
  vector<SlideWindow*> &sws)                           //滑窗结构，用于体素内部管理多帧点
{
  int plsize = pvec->size();
  //遍历当前帧的每一个点
  for(int i=0; i<plsize; i++)
  {
    pointVar &pv = (*pvec)[i];

    //根据世界系坐标计算体素索引
    Eigen::Vector3d &pw = pwld[i];
    float loc[3];
    for(int j=0; j<3; j++)
    {
      loc[j] = pw[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }
    VOXEL_LOC position(loc[0], loc[1], loc[2]);

    //在全局体素地图中查找该体素
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())    //体素已存在
    {
      //step:向已有 OctoTree 中加入当前点
      iter->second->allocate(win_count, pv, pw, sws);
      iter->second->isexist = true;
      if(feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
    }
    else
    {
      OctoTree* ot = new OctoTree(0, wdsize);
      ot->allocate(win_count, pv, pw, sws);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }
  }

}

/**
 * @brief 这是当前帧 → voxel map → OctoTree → SlideWindow的多线程入口函数。
 * note:feat_tem_map为本次滑窗用到的体素
 *
 */
void cut_voxel_multi(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map, int wdsize, PLV(3) &pwld, vector<vector<SlideWindow*>> &sws)
{
  //把属于同一个 voxel 的点索引聚合在一起
  unordered_map<OctoTree*, vector<int>> map_pvec;
  int plsize = pvec->size();

  //遍历当前帧所有点
  for(int i=0; i<plsize; i++)
  {
    // step:计算当前点所属的体素坐标
    // pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    float loc[3];
    for(int j=0; j<3; j++)
    {
      loc[j] = pw[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    // step:查找或创建 OctoTree
    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    OctoTree* ot = nullptr;

    // 体素已经存在
    if(iter != feat_map.end())
    {
      iter->second->isexist = true;
      if(feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
      ot = iter->second;
    }
    else   //新体素
    {
      ot = new OctoTree(0, wdsize);                          //新建一个体素节点（初始化layer=0）
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;   //计算该体素在世界坐标系下的中心位置
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;

      //将新体素加入主地图和本帧用到的voxel集合
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }

    // 将当前点的索引加入该体素对应的点集合，即记录该点属于哪个OctoTree
    map_pvec[ot].push_back(i);
  }

  // step:多线程准备：将 map 转为连续数组
  vector<pair<OctoTree *const, vector<int>>*> octs;
  octs.reserve(map_pvec.size());
  for(auto iter=map_pvec.begin(); iter!=map_pvec.end(); iter++)
    octs.push_back(&(*iter));

  int thd_num = sws.size();                //线程数
  int g_size = octs.size();                //需要处理的体素数量
  if(g_size < thd_num) return;
  vector<thread*> mthreads(thd_num);       //线程指针数组
  double part = 1.0 * g_size / thd_num;    //每个线程负责的体素区间大小

  //从 sws[0] 中平均分配 SlideWindow 指针给各线程
  int swsize = sws[0].size() / thd_num;
  for(int i=1; i<thd_num; i++)
  {
    sws[i].insert(sws[i].end(), sws[0].end() - swsize, sws[0].end());
    sws[0].erase(sws[0].end() - swsize, sws[0].end());
  }

  //启动子线程
  for(int i=1; i<thd_num; i++)
  {
    mthreads[i] = new thread
    (
      [&](int head, int tail, vector<SlideWindow*> &sw)
      {
        //遍历分配给该线程的体素
        for(int j=head; j<tail; j++)
        {
          //遍历该体素中的所有点索引
          for(int k: octs[j]->second)
            octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sw);
        }
      }, part*i, part*(i+1), ref(sws[i])
    );
  }

  for(int i=0; i<thd_num; i++)
  {
    if(i == 0)   //主线程直接注册点到地图
    {
      for(int j=0; j<int(part); j++)
        for(int k: octs[j]->second)
          octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sws[0]);
    }
    else        //确保所有子线程的 OctoTree 的 allocate() 都已经完成
    {
      mthreads[i]->join();
      delete mthreads[i];
    }

  }

}

/**
 * @brief 用于关键帧切分到体素地图
 *
 * @param feat_map
 * @param pvec
 * @param wdsize
 * @param jour
 */
void cut_voxel(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVec &pvec, int wdsize, double jour)
{
  for(pointVar &pv: pvec)
  {
    float loc[3];
    for(int j=0; j<3; j++)
    {
      loc[j] = pv.pnt[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      //note:注意这里如果体素已经存在，向体素中追加历史关键帧点云的时候用的是fix（不需要再估计协方差？）
      iter->second->allocate_fix(pv);
    }
    else
    {
      //体素不存在则新建体素
      OctoTree* ot = new OctoTree(0, wdsize);
      ot->push_fix_novar(pv);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->jour = jour;        //note:标记这个 voxel 最近一次被轨迹覆盖时，机器人已经累计走了 jour 这么远
      feat_map[position] = ot;
    }
  }

}

// Match the point with the plane in the voxel map
int match(unordered_map<VOXEL_LOC, OctoTree*> &feat_map, Eigen::Vector3d &wld, Plane* &pla, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc)
{
  int flag = 0;

  float loc[3];
  for(int j=0; j<3; j++)
  {
    loc[j] = wld[j] / voxel_size;
    if(loc[j] < 0) loc[j] -= 1;
  }
  VOXEL_LOC position(loc[0], loc[1], loc[2]);
  auto iter = feat_map.find(position);
  if(iter != feat_map.end())
  {
    double max_prob = 0;
    flag = iter->second->match(wld, pla, max_prob, var_wld, sigma_d, oc);
    // iter->second->match_end(pv, pla, max_prob);
    if(flag && pla==nullptr)
    {
      printf("pla null max_prob: %lf %ld %ld %ld\n", max_prob, iter->first.x, iter->first.y, iter->first.z);
    }
  }

  return flag;
}

#endif
