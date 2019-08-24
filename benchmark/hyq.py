import sys

import numpy as np
import pinocchio
import crocoddyl
import example_robot_data

WITHDISPLAY = 'disp' in sys.argv
WITHPLOT = 'plot' in sys.argv


def plotSolution(rmodel, xs, us):
    import matplotlib.pyplot as plt
    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    for i in range(nx):
        X[i] = [x[i] for x in xs]
    for i in range(nu):
        U[i] = [u[i] for u in us]

    # Plotting the joint positions, velocities and torques
    plt.figure(1)
    legJointNames = ['HAA', 'HFE', 'KFE']
    # LF foot
    plt.subplot(4, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 10))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
    plt.ylabel('LF')
    plt.legend()

    # LH foot
    plt.subplot(4, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(10, 13))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 9, nq + 12))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
    plt.ylabel('LH')
    plt.legend()

    # RF foot
    plt.subplot(4, 3, 7)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 16))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 8)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 15))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 9)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    plt.ylabel('RF')
    plt.legend()

    # RH foot
    plt.subplot(4, 3, 10)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(16, 19))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 11)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 15, nq + 18))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 12)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
    plt.ylabel('RH')
    plt.legend()
    plt.xlabel('knots')
    plt.show()

    plt.figure(2)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[:rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.show()


class SimpleQuadrupedalGaitProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.lfFootId = self.rmodel.getFrameId(lfFoot)
        self.rfFootId = self.rmodel.getFrameId(rfFoot)
        self.lhFootId = self.rmodel.getFrameId(lhFoot)
        self.rhFootId = self.rmodel.getFrameId(rhFoot)
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros((self.rmodel.nv, 1))])
        self.firstStep = True

    def createCoMProblem(self, x0, comGoTo, timeStep, numKnots):
        """ Create a shooting problem for a CoM forward/backward task.

        :param x0: initial state
        :param comGoTo: initial CoM motion
        :param timeStep: step time for each knot
        :param numKnots: number of knots per each phase
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        # lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        # rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        # lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        # rhFootPos0 = self.rdata.oMf[self.rhFootId].translation

        # Defining the action models along the time instances
        comModels = []

        # Creating the action model for the CoM task
        comForwardModels = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(numKnots)
        ]
        comForwardTermModel = self.createSwingFootModel(timeStep,
                                                        [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                        com0 + np.matrix([comGoTo, 0., 0.]).T)
        comForwardTermModel.differential.costs.costs['comTrack'].weight = 1e6

        comBackwardModels = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(numKnots)
        ]
        comBackwardTermModel = self.createSwingFootModel(timeStep,
                                                         [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                         com0 + np.matrix([-comGoTo, 0., 0.]).T)
        comBackwardTermModel.differential.costs.costs['comTrack'].weight = 1e6

        # Adding the CoM tasks
        comModels += comForwardModels + [comForwardTermModel]
        comModels += comBackwardModels + [comBackwardTermModel]

        # Defining the shooting problem
        problem = crocoddyl.ShootingProblem(x0, comModels, comModels[-1])
        return problem

    def createWalkingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = 0.5325

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)
        ]
        if self.firstStep is True:
            rhStep = self.createFootstepModels(comRef, [rhFootPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfFootId, self.rfFootId, self.lhFootId], [self.rhFootId])
            rfStep = self.createFootstepModels(comRef, [rfFootPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfFootId, self.lhFootId, self.rhFootId], [self.rfFootId])
            self.firstStep = False
        else:
            rhStep = self.createFootstepModels(comRef, [rhFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfFootId, self.rfFootId, self.lhFootId], [self.rhFootId])
            rfStep = self.createFootstepModels(comRef, [rfFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfFootId, self.lhFootId, self.rhFootId], [self.rfFootId])
        lhStep = self.createFootstepModels(comRef, [lhFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                           [self.lfFootId, self.rfFootId, self.rhFootId], [self.lhFootId])
        lfStep = self.createFootstepModels(comRef, [lfFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                           [self.rfFootId, self.lhFootId, self.rhFootId], [self.lfFootId])
        loco3dModel += doubleSupport + rhStep + rfStep
        loco3dModel += doubleSupport + lhStep + lfStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createTrottingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple trotting gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = 0.5325

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)
        ]
        if self.firstStep is True:
            rflhStep = self.createFootstepModels(comRef, [rfFootPos0, lhFootPos0], 0.5 * stepLength, stepHeight,
                                                 timeStep, stepKnots, [self.lfFootId, self.rhFootId],
                                                 [self.rfFootId, self.lhFootId])
            self.firstStep = False
        else:
            rflhStep = self.createFootstepModels(comRef, [rfFootPos0, lhFootPos0], stepLength, stepHeight, timeStep,
                                                 stepKnots, [self.lfFootId, self.rhFootId],
                                                 [self.rfFootId, self.lhFootId])
        lfrhStep = self.createFootstepModels(comRef, [lfFootPos0, rhFootPos0], stepLength, stepHeight, timeStep,
                                             stepKnots, [self.rfFootId, self.lhFootId], [self.lfFootId, self.rhFootId])

        loco3dModel += doubleSupport + rflhStep
        loco3dModel += doubleSupport + lfrhStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createPacingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple pacing gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = 0.5325

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)
        ]
        if self.firstStep is True:
            rightSteps = self.createFootstepModels(comRef, [rfFootPos0, rhFootPos0], 0.5 * stepLength, stepHeight,
                                                   timeStep, stepKnots, [self.lfFootId, self.lhFootId],
                                                   [self.rfFootId, self.rhFootId])
            self.firstStep = False
        else:
            rightSteps = self.createFootstepModels(comRef, [rfFootPos0, rhFootPos0], stepLength, stepHeight, timeStep,
                                                   stepKnots, [self.lfFootId, self.lhFootId],
                                                   [self.rfFootId, self.rhFootId])
        leftSteps = self.createFootstepModels(comRef, [lfFootPos0, lhFootPos0], stepLength, stepHeight, timeStep,
                                              stepKnots, [self.rfFootId, self.rhFootId],
                                              [self.lfFootId, self.lhFootId])

        loco3dModel += doubleSupport + rightSteps
        loco3dModel += doubleSupport + leftSteps

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createBoundingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple bounding gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = 0.5325

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId])
            for k in range(supportKnots)
        ]
        hindSteps = self.createFootstepModels(comRef, [lfFootPos0, rfFootPos0], stepLength, stepHeight, timeStep,
                                              stepKnots, [self.lhFootId, self.rhFootId],
                                              [self.lfFootId, self.rfFootId])
        frontSteps = self.createFootstepModels(comRef, [lhFootPos0, rhFootPos0], stepLength, stepHeight, timeStep,
                                               stepKnots, [self.lfFootId, self.rfFootId],
                                               [self.lhFootId, self.rhFootId])

        loco3dModel += doubleSupport + hindSteps
        loco3dModel += doubleSupport + frontSteps

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createJumpingProblem(self, x0, jumpHeight, timeStep):
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = 0.5325

        takeOffKnots = 30
        flyingKnots = 30

        loco3dModel = []
        takeOff = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(takeOffKnots)
        ]
        flyingPhase = [
            self.createSwingFootModel(timeStep, [],
                                      np.matrix([0., 0., jumpHeight * (k + 1) / flyingKnots]).T + comRef)
            for k in range(flyingKnots)
        ]

        loco3dModel += takeOff
        loco3dModel += flyingPhase

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createFootstepModels(self, comPos0, feetPos0, stepLength, stepHeight, timeStep, numKnots, supportFootIds,
                             swingFootIds):
        """ Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs

        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length
                # resKnot = numKnots % 2
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = np.matrix([[stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots]]).T
                elif k == phKnots:
                    dp = np.matrix([[stepLength * (k + 1) / numKnots, 0., stepHeight]]).T
                else:
                    dp = np.matrix(
                        [[stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)]]).T
                tref = np.asmatrix(p + dp)

                swingFootTask += [crocoddyl.FramePlacement(i, pinocchio.SE3(np.eye(3), tref))]

            # Adding an action model for this knot
            comTask = np.matrix([stepLength * (k + 1) / numKnots, 0., 0.]).T * comPercentage + comPos0
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
            ]
        # Action model for the foot switch
        footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask)

        # Updating the current foot position for next step
        comPos0 += np.matrix([stepLength * comPercentage, 0., 0.]).T
        for p in feetPos0:
            p += np.matrix([[stepLength, 0., 0.]]).T
        return footSwingModel + [footSwitchModel]

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            xref = crocoddyl.FrameTranslation(i, np.matrix([0., 0., 0.]).T)
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, np.matrix([0., 0.]).T)
            contactModel.addContact('contact_' + str(i), supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if isinstance(comTask, np.ndarray):
            comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1e4)
        if swingFootTask is not None:
            for i in swingFootTask:
                xref = crocoddyl.FrameTranslation(i.frame, i.oMf.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                costModel.addCost("footTrack_" + str(i), footTrack, 1e4)
        stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(np.matrix(stateWeights**2).T),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e-1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-4)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createFootSwitchModel(self, supportFootIds, swingFootTask):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return action model for a foot switch phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            xref = crocoddyl.FrameTranslation(i, np.matrix([0., 0., 0.]).T)
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, np.matrix([0., 0.]).T)
            contactModel.addContact('contact_' + str(i), supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if swingFootTask is not None:
            for i in swingFootTask:
                xref = crocoddyl.FrameTranslation(i.frame, i.oMf.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                costModel.addCost("footTrack_" + str(i), footTrack, 1e7)
        stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(np.matrix(stateWeights**2).T),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        for i in swingFootTask:
            vref = crocoddyl.FrameMotion(i.frame, pinocchio.Motion.Zero())
            impactFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            costModel.addCost('impactVel_' + str(i.frame), impactFootVelCost, 1e6)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        return model


# Loading the HyQ model
hyq = example_robot_data.loadHyQ()

rmodel = hyq.model

# Defining the initial state of the robot
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
v0 = pinocchio.utils.zero(rmodel.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
lfFoot = 'lf_foot'
rfFoot = 'rf_foot'
lhFoot = 'lh_foot'
rhFoot = 'rh_foot'
gait = SimpleQuadrupedalGaitProblem(rmodel, lfFoot, rfFoot, lhFoot, rhFoot)

# Setting up all tasks
GAITPHASES = [{
    'walking': {
        'stepLength': 0.15,
        'stepHeight': 0.2,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 5
    }
}, {
    'trotting': {
        'stepLength': 0.15,
        'stepHeight': 0.2,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 5
    }
}, {
    'pacing': {
        'stepLength': 0.15,
        'stepHeight': 0.2,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 5
    }
}, {
    'bounding': {
        'stepLength': 0.15,
        'stepHeight': 0.2,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 5
    }
}, {
    'jumping': {
        'jumpHeight': 0.5,
        'timeStep': 1e-2
    }
}]
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots']))
        elif key == 'trotting':
            # Creating a trotting problem
            ddp[i] = crocoddyl.SolverDDP(
                gait.createTrottingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                           value['stepKnots'], value['supportKnots']))
        elif key == 'pacing':
            # Creating a pacing problem
            ddp[i] = crocoddyl.SolverDDP(
                gait.createPacingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                         value['stepKnots'], value['supportKnots']))
        elif key == 'bounding':
            # Creating a bounding problem
            ddp[i] = crocoddyl.SolverDDP(
                gait.createBoundingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                           value['stepKnots'], value['supportKnots']))
        elif key == 'jumping':
            # Creating a jumping problem
            ddp[i] = crocoddyl.SolverDDP(gait.createJumpingProblem(x0, value['jumpHeight'], value['timeStep']))

    # Added the callback functions
    print('*** SOLVE ' + key + ' ***')
    if WITHDISPLAY:
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackSolverDisplay(hyq, 4, 4, cameraTF)])
    else:
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    ddp[i].th_stop = 1e-9
    xs = [rmodel.defaultState] * len(ddp[i].models())
    us = [m.quasicStatic(d, rmodel.defaultState) for m, d in list(zip(ddp[i].models(), ddp[i].datas()))[:-1]]
    ddp[i].solve(xs, us, 100, False, 0.1)

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    for i, phase in enumerate(GAITPHASES):
        crocoddyl.displayTrajectory(hyq, ddp[i].xs, ddp[i].models()[0].dt)

# Plotting the entire motion
if WITHPLOT:
    xs = []
    us = []
    for i, phase in enumerate(GAITPHASES):
        xs.extend(ddp[i].xs)
        us.extend(ddp[i].us)
    plotSolution(rmodel, xs, us)
