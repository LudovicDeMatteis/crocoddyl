///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "solver.hpp"

#include "crocoddyl/core/solvers/box-ddp.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#ifdef CROCODDYL_WITH_IPOPT
#include "crocoddyl/core/solvers/ipopt.hpp"
#endif
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<SolverTypes::Type> SolverTypes::all(SolverTypes::init_all());

std::ostream& operator<<(std::ostream& os, SolverTypes::Type type) {
  switch (type) {
    case SolverTypes::SolverKKT:
      os << "SolverKKT";
      break;
    case SolverTypes::SolverDDP:
      os << "SolverDDP";
      break;
    case SolverTypes::SolverFDDP:
      os << "SolverFDDP";
      break;
    case SolverTypes::SolverBoxDDP:
      os << "SolverBoxDDP";
      break;
    case SolverTypes::SolverBoxFDDP:
      os << "SolverBoxFDDP";
      break;
#ifdef CROCODDYL_WITH_IPOPT
    case SolverTypes::SolverIpopt:
      os << "SolverIpopt";
      break;
#endif
    case SolverTypes::NbSolverTypes:
      os << "NbSolverTypes";
      break;
    default:
      break;
  }
  return os;
}

SolverFactory::SolverFactory() {}

SolverFactory::~SolverFactory() {}

boost::shared_ptr<crocoddyl::SolverAbstract> SolverFactory::create(
    SolverTypes::Type solver_type,
    boost::shared_ptr<crocoddyl::ActionModelAbstract> model,
    boost::shared_ptr<crocoddyl::ActionModelAbstract> model2,
    boost::shared_ptr<crocoddyl::ActionModelAbstract> modelT, size_t T) const {
  boost::shared_ptr<crocoddyl::SolverAbstract> solver;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >
      running_models;
  const size_t halfway = T / 2;
  for (size_t i = 0; i < halfway; ++i) {
    running_models.push_back(model);
  }
  for (size_t i = 0; i < T - halfway; ++i) {
    running_models.push_back(model2);
  }

  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(model->get_state()->zero(),
                                                     running_models, modelT);

  switch (solver_type) {
    case SolverTypes::SolverKKT:
      solver = boost::make_shared<crocoddyl::SolverKKT>(problem);
      break;
    case SolverTypes::SolverDDP:
      solver = boost::make_shared<crocoddyl::SolverDDP>(problem);
      break;
    case SolverTypes::SolverFDDP:
      solver = boost::make_shared<crocoddyl::SolverFDDP>(problem);
      break;
    case SolverTypes::SolverBoxDDP:
      solver = boost::make_shared<crocoddyl::SolverBoxDDP>(problem);
      break;
    case SolverTypes::SolverBoxFDDP:
      solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem);
      break;
#ifdef CROCODDYL_WITH_IPOPT
    case SolverTypes::SolverIpopt:
      solver = boost::make_shared<crocoddyl::SolverIpopt>(problem);
      break;
#endif
    default:
      throw_pretty(__FILE__ ": Wrong SolverTypes::Type given");
      break;
  }
  return solver;
}

}  // namespace unittest
}  // namespace crocoddyl
