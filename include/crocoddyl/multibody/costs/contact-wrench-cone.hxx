///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameWrenchCone& fref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone, nu)),
      fref_(fref) {
  std::cerr << "Deprecated CostModelContactWrenchCone: Use ResidualModelContactWrenchCone with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameWrenchCone& fref)
    : Base(state, activation, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone)),
      fref_(fref) {
  std::cerr << "Deprecated CostModelContactWrenchCone:esidualModelContactWrenchCone with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const FrameWrenchCone& fref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone, nu)), fref_(fref) {
  std::cerr << "Deprecated CostModelContactWrenchCone: Use ResidualModelContactWrenchCone with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const FrameWrenchCone& fref)
    : Base(state, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone)), fref_(fref) {
  std::cerr << "Deprecated CostModelContactWrenchCone: Use ResidualModelContactWrenchCone with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelContactWrenchConeTpl<Scalar>::~CostModelContactWrenchConeTpl() {}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference contact wrench
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x,
                                                     const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and contact wrench cone residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->residual->Ru.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  d->Arr_Ru.noalias() = data->activation->Arr * data->residual->Ru;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
  data->Lxu.noalias() = data->residual->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->residual->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactWrenchConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameWrenchCone)) {
    fref_ = *static_cast<const FrameWrenchCone*>(pv);
    ResidualModelContactWrenchCone* residual = static_cast<ResidualModelContactWrenchCone*>(residual_.get());
    residual->set_id(fref_.id);
    residual->set_reference(fref_.cone);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

template <typename Scalar>
void CostModelContactWrenchConeTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameWrenchCone)) {
    FrameWrenchCone& ref_map = *static_cast<FrameWrenchCone*>(pv);
    ResidualModelContactWrenchCone* residual = static_cast<ResidualModelContactWrenchCone*>(residual_.get());
    fref_.id = residual->get_id();
    fref_.cone = residual->get_reference();
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

}  // namespace crocoddyl
