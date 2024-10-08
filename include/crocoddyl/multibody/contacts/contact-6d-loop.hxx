
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"

namespace crocoddyl {

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::ContactModel6DLoopTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex frame1_id,
    const pinocchio::FrameIndex frame2_id, const pinocchio::ReferenceFrame ref, 
    const std::size_t nu, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 6, nu), gains_(gains), 
    frame1_id_(frame1_id), frame2_id_(frame2_id), is_frame_(true) {
  if(ref != pinocchio::ReferenceFrame::LOCAL){
    std::cerr << "Warning: Only reference frame LOCAL is supported for 6D loop contacts\n";
  }
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::ContactModel6DLoopTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex frame1_id,
    const pinocchio::FrameIndex frame2_id, const pinocchio::ReferenceFrame ref, 
    const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 6), gains_(gains),
    joint1_id_(frame1_id), frame2_id_(frame2_id), is_frame_(true) {
  if(ref != pinocchio::ReferenceFrame::LOCAL){
    std::cerr << "Warning: Only reference frame LOCAL is supported for 6D loop contacts\n";
  }
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::ContactModel6DLoopTpl(
    boost::shared_ptr<StateMultibody> state, 
    const int joint1_id, const SE3& joint1_placement,
    const int joint2_id, const SE3& joint2_placement,
    const pinocchio::ReferenceFrame ref, const std::size_t nu, const Vector2s& gains)
    : Base(state, pinocchio::ReferenceFrame::LOCAL, 6, nu),
      gains_(gains), joint1_id_(joint1_id), joint2_id_(joint2_id),
      joint1_placement_(joint1_placement), joint2_placement_(joint2_placement), is_frame_(false) {
}

template <typename Scalar>
ContactModel6DLoopTpl<Scalar>::~ContactModel6DLoopTpl() {}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calc(
    const boost::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacements<Scalar>(*state_->get_pinocchio().get(),
                                          *d->pinocchio);
  d->j1Xf1 = joint1_placement_.toActionMatrix();
  d->j2Xf2 = joint2_placement_.toActionMatrix();  

  pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                            joint1_id_, pinocchio::LOCAL, d->j1Jj1);
  pinocchio::getJointJacobian(*state_->get_pinocchio().get(), *d->pinocchio,
                            joint2_id_, pinocchio::LOCAL, d->j2Jj2);
  d->f1Jf1 = d->j1Xf1.inverse() * d->j1Jj1;
  d->f2Jf2 = d->j2Xf2.inverse() * d->j2Jj2;

  SE3 oMf1 = d->pinocchio->oMi[joint1_id_].act(joint1_placement_);
  SE3 oMf2 = d->pinocchio->oMi[joint2_id_].act(joint2_placement_);
  d->f1Mf2 = oMf1.actInv(oMf2);
  d->f1Xf2 = d->f1Mf2.toActionMatrix();

  d->Jc = d->f1Jf1 - d->f1Xf2 * d->f2Jf2;
  // Compute the acceleration drift
  if(joint1_id_ > 0){
    d->f1vf1 = joint1_placement_.actInv(d->pinocchio->v[joint1_id_]);
    d->f1af1 = joint1_placement_.actInv(d->pinocchio->a[joint1_id_]);
  }
  else{
    d->f1vf1.setZero();
    d->f1af1.setZero();
  }
  if(joint2_id_ > 0){
    d->f2vf2 = joint2_placement_.actInv(d->pinocchio->v[joint2_id_]);
    d->f2af2 = joint2_placement_.actInv(d->pinocchio->a[joint2_id_]);
  }
  else{
    d->f2vf2.setZero();
    d->f2af2.setZero();
  }
  d->f1vf2 = d->f1Mf2.act(d->f2vf2);
  d->f1af2 = d->f1Mf2.act(d->f2af2);
  d->a0 = (d->f1af1 - d->f1Mf2.act(d->f2af2) + d->f1vf1.cross(d->f1vf2)).toVector();
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  if(gains_[0] != 0. || gains_[1] != 0.){
    throw_pretty("Baumgarte stabilization is not implemented yet");
  }
  
  if(joint1_id_ > 0){
    d->f1af1 = joint1_placement_.actInv(d->pinocchio->a[joint1_id_]);
  }
  else{
    d->f1af1.setZero();
  }
  if(joint2_id_ > 0){
    d->f2af2 = joint2_placement_.actInv(d->pinocchio->a[joint2_id_]);
  }
  else{
    d->f2af2.setZero();
  }
  d->f1af2 = d->f1Mf2.act(d->f2af2);

  const std::size_t nv = state_->get_nv();
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint1_id_, pinocchio::LOCAL,
      d->v1_partial_dq, d->a1_partial_dq, d->a1_partial_dv, d->a1_partial_da);
  pinocchio::getJointAccelerationDerivatives(
      *state_->get_pinocchio().get(), *d->pinocchio, joint2_id_, pinocchio::LOCAL,
      d->v2_partial_dq, d->a2_partial_dq, d->a2_partial_dv, d->a2_partial_da);

  d->da0_dq_t1 = joint1_placement_.toActionMatrixInverse() * d->a1_partial_dq; //
  d->da0_dq_t2 = (d->f1af2.toActionMatrix() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2) + d->f1Xf2 * (joint2_placement_.toActionMatrixInverse()*d->a2_partial_dq)); //
  d->da0_dq_t3 = - d->f1vf2.toActionMatrix() * (joint1_placement_.toActionMatrixInverse() * d->v1_partial_dq) // part 1
                + d->f1vf1.toActionMatrix() * d->f1vf2.toActionMatrix() * (d->f1Jf1 - d->f1Xf2 * d->f2Jf2)            // part 2
                + d->f1vf1.toActionMatrix() * d->f1Xf2 * (joint2_placement_.toActionMatrixInverse() * d->v2_partial_dq);     // part 3


  d->da0_dx.leftCols(nv) = d->da0_dq_t1 - d->da0_dq_t2 + d->da0_dq_t3; // This should be da0_dq
  d->da0_dx.rightCols(nv) = joint1_placement_.toActionMatrixInverse() * d->a1_partial_dv
                            - d->f1Xf2 * (joint2_placement_.toActionMatrixInverse() * d->a2_partial_dv) 
                            - d->f1vf2.toActionMatrix() * d->f1Jf1
                            + d->f1vf1.toActionMatrix() * d->f1Xf2 * d->f2Jf2; // This should be da0_dv
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::updateForce(
    const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 6) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 6)");
  }
  Data* d = static_cast<Data*>(data.get());
  d->f = pinocchio::ForceTpl<Scalar>(-force);
  switch(type_){
    case pinocchio::ReferenceFrame::LOCAL:
    {
      data->fext = joint1_placement_.act(data->f);
      d->joint1_f = - joint1_placement_.act(data->f);
      d->joint2_f = (joint2_placement_ * d->f1Mf2.inverse()).act(data->f);

      data->dtau_dq.setZero();
      
      d->f_cross.setZero();
      d->f_cross.topRightCorner(3,3) = pinocchio::skew(d->joint2_f.linear());
      d->f_cross.bottomLeftCorner(3,3) = pinocchio::skew(d->joint2_f.linear());
      d->f_cross.bottomRightCorner(3,3) = pinocchio::skew(d->joint2_f.angular());

      SE3 j2Mj1 = joint2_placement_.act(d->f1Mf2.actInv(joint1_placement_.inverse()));

      data->dtau_dq = d->j2Jj2.transpose() * (- d->f_cross * (d->j2Jj2 - j2Mj1.toActionMatrix()*d->j1Jj1));
      break;
    }
    case pinocchio::ReferenceFrame::WORLD:
      throw_pretty("Reference frame WORLD is not implemented yet");
      break;
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      throw_pretty("Reference frame LOCAL_WORLD_ALIGNED is not implemented yet");
      break;
  }
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::updateForceDiff(
    const boost::shared_ptr<ContactDataAbstract>& data, const MatrixXs& df_dx,
    const MatrixXs& df_du) const {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ ||
      static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx())
    throw_pretty("df_dx has wrong dimension");

  if (static_cast<std::size_t>(df_du.rows()) != nc_ ||
      static_cast<std::size_t>(df_du.cols()) != nu_)
    throw_pretty("df_du has wrong dimension");

  data->df_dx = -df_dx;
  data->df_du = -df_du;
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModel6DLoopTpl<Scalar>::createData(pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ContactModel6DLoopTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel6D {frame=" << state_->get_pinocchio()->frames[id_].name
     << ", type=" << type_ << "}";
}


template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s&
ContactModel6DLoopTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
const int ContactModel6DLoopTpl<Scalar>::get_joint1_id() const {
  return joint1_id_;
}

template <typename Scalar>
const int ContactModel6DLoopTpl<Scalar>::get_joint2_id() const {
  return joint2_id_;
}

template <typename Scalar>
const typename pinocchio::SE3Tpl<Scalar>&
ContactModel6DLoopTpl<Scalar>::get_joint1_placement() const {
  return joint1_placement_;
}

template <typename Scalar>
const typename pinocchio::SE3Tpl<Scalar>&
ContactModel6DLoopTpl<Scalar>::get_joint2_placement() const {
  return joint2_placement_;
}

}  // namespace crocoddyl
