///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
#define CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/force.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ImpulseModelAbstractTpl {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ImpulseDataAbstractTpl<Scalar> ImpulseDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ImpulseModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& ni);
  virtual ~ImpulseModelAbstractTpl();

  virtual void calc(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;

  virtual void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const VectorXs& force) = 0;
  void updateForceDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const MatrixXs& df_dq) const;

  virtual boost::shared_ptr<ImpulseDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const std::size_t& get_ni() const;

 protected:
  boost::shared_ptr<StateMultibody> state_;
  std::size_t ni_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ImpulseDataAbstract>& data, const VectorXs& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ImpulseDataAbstract>& data, const VectorXs& x) { calcDiff(data, x); }

#endif
};

template <typename _Scalar>
struct ImpulseDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ImpulseDataAbstractTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : pinocchio(data),
        joint(0),
        frame(0),
        Jc(model->get_ni(), model->get_state()->get_nv()),
        dv0_dq(model->get_ni(), model->get_state()->get_nv()),
        f(pinocchio::ForceTpl<Scalar>::Zero()),
        df_dq(model->get_ni(), model->get_state()->get_nv()) {
    Jc.setZero();
    dv0_dq.setZero();
    df_dq.setZero();
  }
  virtual ~ImpulseDataAbstractTpl() {}

  pinocchio::DataTpl<Scalar>* pinocchio;
  pinocchio::JointIndex joint;
  pinocchio::FrameIndex frame;
  MatrixXs Jc;
  MatrixXs dv0_dq;
  pinocchio::ForceTpl<Scalar> f;
  MatrixXs df_dq;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/impulse-base.hxx"

#endif  // CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
