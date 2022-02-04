import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import itk
from scipy.spatial.transform import Rotation as R

class SlicerJupyterServerHelper:
  def installRequiredPackages(self, force=False):
    # Need to install if forced or any packages cannot be imported
    needToInstall = force
    if not needToInstall:
      try:
        import jupyter
        import ipywidgets
        import pandas
        import ipyevents
        import ipycanvas
      except:
        needToInstall = True

    if needToInstall:
      # Install required packages
      import os
      if os.name == 'nt':
        # There are no official pyzmq wheels for Python-3.6 for Windows, so we have to install manually
        slicer.util.pip_install(
          "https://files.pythonhosted.org/packages/94/e1/13059383d21444caa16306b48c8bf7a62331ca361d553d2119696ea67119/pyzmq-19.0.0-cp36-cp36m-win_amd64.whl")
      else:
        # PIL may be corrupted on linux, reinstall from pillow
        slicer.util.pip_install('--upgrade pillow --force-reinstall')

      slicer.util.pip_install("jupyter ipywidgets pandas ipyevents ipycanvas --no-warn-script-location")
      slicer.util._executePythonModule("jupyter", "nbextension enable --py widgetsnbextension".split(" "))
      slicer.util._executePythonModule("jupyter", "nbextension enable --py ipyevents".split(" "))

    # Install Slicer Jupyter kernel
    # Create Slicer kernel
    slicer.modules.jupyterkernel.updateKernelSpec()
    # Install Slicer kernel
    import jupyter_client
    jupyter_client.kernelspec.KernelSpecManager().install_kernel_spec(slicer.modules.jupyterkernel.kernelSpecPath(),
                                                                    user=True, replace=True)


#
# XraySimulation
#

class SNSClinicalSimulation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SNSClinicalSimulation"  # TODO make this more human readable by adding spaces
    self.parent.categories = ["Sacral Nerve Stimulation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Rafael Moreta-Martinez (Universidad Carlos III de Madrid), Monica García-Sevilla (Universidad Carlos III de Madrid)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This simulates the sacral neurostimulation surgery.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Rafael Moreta Martinez, Universidad Carlos III de Madrid, during his PhD.
"""

#
# SNSClinicalSimulationWidget
#

class SNSClinicalSimulationWidget(ScriptedLoadableModuleWidget):


  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate logic
    self.logic = SNSClinicalSimulationLogic()

    # Layout setup: 3D Only
    self.layoutManager = slicer.app.layoutManager()
    self.setCustomLayout()
    self.layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

    # Init Variables
    self.connect = True

    self.modeSelected = "1"
    self.logic.modeSelected = self.modeSelected
    self.numberOfPunctures = 0
    self.userID, self.repetitionID = "None", "None"
    self.timesTargetReachedButtonClicked = 0
    self.targetSelected = "None"

    # Style definition
    self.defaultStyleSheet = "QLabel { color : #000000; \
                                          font: bold 14px}"
    self.collapsibleButtonStyleSheet = "font-size: 16px; font-weight: bold; "
    self.groupBoxStyleSheet = "font-size: 14px"
    self.pushButtonStyleSheet = "font-size: 12px; font-weight: bold; background-color: rgb(222,222,222)"
    self.qLineStyleSheet = "font-size: 14px; background-color: rgb(222,222,222);"


    ################################################################################
    ############################## Initialization ##################################
    ################################################################################

    self.initCollapsibleButton = ctk.ctkCollapsibleButton()
    self.initCollapsibleButton.text = "INITIALIZATION"
    self.layout.addWidget(self.initCollapsibleButton)
    initFormLayout = qt.QFormLayout(self.initCollapsibleButton)

    ## Init Layout Button
    self.initViewPointButton = qt.QPushButton("3D VIEW") # text in button
    self.initViewPointButton.enabled = False
    initFormLayout.addRow(self.initViewPointButton)

    #---------------------------#
    #------ Connections --------#
    #---------------------------#
    self.connections_GroupBox = ctk.ctkCollapsibleGroupBox()
    self.connections_GroupBox.setTitle("Connections")
    self.connections_GroupBox.collapsed = False
    initFormLayout.addRow(self.connections_GroupBox)

    connections_GroupBox_Layout = qt.QFormLayout(self.connections_GroupBox)
    connections_GroupBox_H_Layout = qt.QHBoxLayout()
    connections_GroupBox_Layout.addRow(connections_GroupBox_H_Layout)

    arrangeConnectButtons_Layout = qt.QHBoxLayout()
    connections_GroupBox_Layout.addRow(arrangeConnectButtons_Layout)

    ## Connection to PLUS
    self.connectToPlusButton = qt.QPushButton("Connect to Plus")
    self.connectToPlusButton.enabled = True
    arrangeConnectButtons_Layout.addWidget(self.connectToPlusButton)

    ## Load data
    self.loadDataButton = qt.QPushButton("Load Data")
    self.loadDataButton.enabled = True
    arrangeConnectButtons_Layout.addWidget(self.loadDataButton)

    self.doneWithInitButton = qt.QPushButton("DONE WITH INIT")
    self.doneWithInitButton.enabled = True
    initFormLayout.addRow(self.doneWithInitButton)

    ################################################################################
    ############################### Simulation #####################################
    ################################################################################

    self.simulationCollapsibleButton = ctk.ctkCollapsibleButton()
    self.simulationCollapsibleButton.text = "SIMULATION"
    self.simulationCollapsibleButton.collapsed = True
    self.layout.addWidget(self.simulationCollapsibleButton)
    simulationFormLayout = qt.QFormLayout(self.simulationCollapsibleButton)


    viewMode_H_Layout = qt.QHBoxLayout()
    simulationFormLayout.addRow(viewMode_H_Layout)
    ## Simulation View Point Button
    self.simulationViewPointButton = qt.QPushButton("SIMULATION VIEW")
    self.simulationViewPointButton.enabled = True
    viewMode_H_Layout.addWidget(self.simulationViewPointButton)

    ## Init View Point Button
    self.initViewPointButton2 = qt.QPushButton("3D VIEW")
    self.initViewPointButton2.enabled = False
    viewMode_H_Layout.addWidget(self.initViewPointButton2)

    #---------------------------#
    #-------- User INFO --------#
    #---------------------------#
    self.userInformation_GroupBox = ctk.ctkCollapsibleGroupBox()
    self.userInformation_GroupBox.setTitle("Target")  # title for collapsible button
    self.userInformation_GroupBox.collapsed = False  # if True it appears collapsed
    simulationFormLayout.addRow(self.userInformation_GroupBox)  # add collapsible button to layout

    userInformation_GroupBox_Layout = qt.QFormLayout(self.userInformation_GroupBox)

    targetSelection_H_Layout = qt.QHBoxLayout()
    userInformation_GroupBox_Layout.addRow(targetSelection_H_Layout)
    ## None mode radio button
    self.modeNone_radioButton = qt.QRadioButton('None')
    self.modeNone_radioButton.checked = True
    targetSelection_H_Layout.addWidget(self.modeNone_radioButton)

    ## S3L
    self.modeS3L_radioButton = qt.QRadioButton('S3L')
    targetSelection_H_Layout.addWidget(self.modeS3L_radioButton)
    ## S3R
    self.modeS3R_radioButton = qt.QRadioButton('S3R')
    targetSelection_H_Layout.addWidget(self.modeS3R_radioButton)
    ## S4L
    self.modeS4L_radioButton = qt.QRadioButton('S4L')
    targetSelection_H_Layout.addWidget(self.modeS4L_radioButton)
    ## S4R
    self.modeS4R_radioButton = qt.QRadioButton('S4R')
    targetSelection_H_Layout.addWidget(self.modeS4R_radioButton)

    #---------------------------#
    #------- Simulate DRR ------#
    #---------------------------#
    self.simulationDRR_GroupBox = ctk.ctkCollapsibleGroupBox()
    self.simulationDRR_GroupBox.setTitle("DRR Simulation")
    self.simulationDRR_GroupBox.enabled = True
    self.simulationDRR_GroupBox.setStyleSheet("background-color: rgb(189,227,254);")
    self.simulationDRR_GroupBox.collapsed = False
    simulationFormLayout.addRow(self.simulationDRR_GroupBox)
    simulationDRR_GroupBox_Layout = qt.QFormLayout(self.simulationDRR_GroupBox)
    simulationDRR_H_Layout = qt.QHBoxLayout()
    simulationDRR_GroupBox_Layout.addRow(simulationDRR_H_Layout)

    ##---- START SIMULATION ----##
    ## Start Simulation Button
    self.startSimulationRepetitionButton = qt.QPushButton("START")
    self.startSimulationRepetitionButton.setStyleSheet(self.pushButtonStyleSheet)
    self.startSimulationRepetitionButton.enabled = False
    simulationDRR_GroupBox_Layout.addRow(self.startSimulationRepetitionButton)

    ##------------------------------------------------
    ##---------- DRR PARAMS -------------##
    self.DRRParams_GroupBox = ctk.ctkCollapsibleGroupBox()
    self.DRRParams_GroupBox.setTitle("DRR Params")
    self.DRRParams_GroupBox.collapsed = True
    simulationDRR_GroupBox_Layout.addRow(self.DRRParams_GroupBox)
    DRRParams_GroupBox_Layout = qt.QFormLayout(self.DRRParams_GroupBox)

    DRRParamsRow1_Layout = qt.QHBoxLayout()
    DRRParams_GroupBox_Layout.addRow(DRRParamsRow1_Layout)
    ## Focal point
    self.focalPointSpinBox = ctk.ctkDoubleSpinBox()
    self.focalPointSpinBox.maximum = 1000
    self.focalPointSpinBox.minimum = 0
    self.focalPointSpinBox.value = 800
    self.focalPointSpinBox.decimals = 0
    self.focalPointSpinBox.singleStep = 1

    focalPointInfoText = qt.QLabel("\tFocal Point: ")
    DRRParamsRow1_Layout.addWidget(focalPointInfoText)
    DRRParamsRow1_Layout.addWidget(self.focalPointSpinBox)

    ## DRR Threshold
    self.drrThresholdSpinBox = ctk.ctkDoubleSpinBox()
    self.drrThresholdSpinBox.maximum = 1000
    self.drrThresholdSpinBox.minimum = -1000
    self.drrThresholdSpinBox.value = 50
    self.drrThresholdSpinBox.decimals = 0
    self.drrThresholdSpinBox.singleStep = 1

    focalPointInfoText = qt.QLabel("\tThreshold: ")
    DRRParamsRow1_Layout.addWidget(focalPointInfoText)
    DRRParamsRow1_Layout.addWidget(self.drrThresholdSpinBox)

    DRRParamsRow2_Layout = qt.QHBoxLayout()
    DRRParams_GroupBox_Layout.addRow(DRRParamsRow2_Layout)
    ## DRR Size Text
    focalPointInfoText = qt.QLabel("DRR Size: ")
    DRRParamsRow2_Layout.addWidget(focalPointInfoText)

    ## DRR Size X
    self.drrSizeXSpinBox = ctk.ctkDoubleSpinBox()
    self.drrSizeXSpinBox.maximum = 1024
    self.drrSizeXSpinBox.minimum = 1
    self.drrSizeXSpinBox.value = 512
    self.drrSizeXSpinBox.decimals = 0
    self.drrSizeXSpinBox.singleStep = 1
    self.drrSizeXSpinBox.minimum = 0

    focalPointInfoText = qt.QLabel("\tX: ")
    DRRParamsRow2_Layout.addWidget(focalPointInfoText)
    DRRParamsRow2_Layout.addWidget(self.drrSizeXSpinBox)

    ## DRR Size X
    self.drrSizeYSpinBox = ctk.ctkDoubleSpinBox()
    self.drrSizeYSpinBox.maximum = 1024
    self.drrSizeYSpinBox.minimum = 1
    self.drrSizeYSpinBox.value = 512
    self.drrSizeYSpinBox.decimals = 0
    self.drrSizeYSpinBox.singleStep = 1
    self.drrSizeYSpinBox.minimum = 0

    focalPointInfoText = qt.QLabel("\tY: ")
    DRRParamsRow2_Layout.addWidget(focalPointInfoText)
    DRRParamsRow2_Layout.addWidget(self.drrSizeYSpinBox)

    ##------------------------------------------------
    ##---------- MAKE PROJECTION MODE 1 -------------##
    self.makeProjectionMode1_GroupBox = ctk.ctkCollapsibleGroupBox()
    self.makeProjectionMode1_GroupBox.setTitle("Mode 1")
    self.makeProjectionMode1_GroupBox.collapsed = False
    self.makeProjectionMode1_GroupBox.enabled = False
    simulationDRR_GroupBox_Layout.addRow(self.makeProjectionMode1_GroupBox)
    makeProjectionMode1_GroupBox_Layout = qt.QFormLayout(self.makeProjectionMode1_GroupBox)


    self.projectionButtonStyleSheet = "font-size: 14px; font-weight: bold; background-color: rgb(222,222,222)"


    makeProjectionsButtonsMode1_Layout = qt.QHBoxLayout()
    makeProjectionMode1_GroupBox_Layout.addRow(makeProjectionsButtonsMode1_Layout)
    ## Make Lateral Projection Button
    self.makeLateralProjectionButton = qt.QPushButton("LATERAL\nPROJECTION")
    self.makeLateralProjectionButton.setStyleSheet(self.projectionButtonStyleSheet)
    self.makeLateralProjectionButton.enabled = True
    makeProjectionsButtonsMode1_Layout.addWidget(self.makeLateralProjectionButton)

    ## Make Anterior Projection Button
    self.makeAnteriorProjectionButton = qt.QPushButton("ANTERIOR\nPROJECTION")
    self.makeAnteriorProjectionButton.setStyleSheet(self.projectionButtonStyleSheet)
    self.makeAnteriorProjectionButton.enabled = True
    makeProjectionsButtonsMode1_Layout.addWidget(self.makeAnteriorProjectionButton)


    ##--------------------------------------------------
    ##---------- ADD NUMBER OF PUNCTURES -------------##

    DRR_addNumberOfPunctures_Layout = qt.QHBoxLayout()
    simulationDRR_GroupBox_Layout.addRow(DRR_addNumberOfPunctures_Layout)
    ## Add Puncture Button
    self.DRRAddPunctureButton = qt.QPushButton("Add Puncture")
    self.DRRAddPunctureButton.enabled = False
    DRR_addNumberOfPunctures_Layout.addWidget(self.DRRAddPunctureButton)

    ## Remove Puncture Button
    self.DRRRemovePunctureButton = qt.QPushButton("Remove Puncture")
    self.DRRRemovePunctureButton.enabled = False
    DRR_addNumberOfPunctures_Layout.addWidget(self.DRRRemovePunctureButton)

    ## Number of Puncters Infotext
    self.DRRNumberOfPunctures_InfoText = qt.QLabel('Number Of Punctures: 0')
    DRR_addNumberOfPunctures_Layout.addWidget(self.DRRNumberOfPunctures_InfoText)

    ##--------------------------------------------------
    ##---------- CHECK IF TARGET HAS BEEN REACHED -------------##
    self.targetReached_GroupBox = ctk.ctkCollapsibleGroupBox()
    self.targetReached_GroupBox.setTitle("Target Reached")
    self.targetReached_GroupBox.collapsed = False
    self.targetReached_GroupBox.enabled = True
    simulationDRR_GroupBox_Layout.addRow(self.targetReached_GroupBox)
    targetReached_GroupBox_Layout = qt.QFormLayout(self.targetReached_GroupBox)


    DRR_targetReached_Layout = qt.QHBoxLayout()
    targetReached_GroupBox_Layout.addRow(DRR_targetReached_Layout)

    ## Check if user has reached Target Button
    self.targetReachedButton = qt.QPushButton("CHECK IF TARGET REACHED")
    self.targetReachedButton.setStyleSheet(self.pushButtonStyleSheet)
    self.targetReachedButton.enabled = False
    DRR_targetReached_Layout.addWidget(self.targetReachedButton)

    ## Text
    self.targetReachedInfoText = qt.QLabel('-')
    self.targetReachedStyleSheet = "font-size: 16px; font-weight: bold; "
    self.targetReachedInfoText.setStyleSheet(self.targetReachedStyleSheet)
    DRR_targetReached_Layout.addWidget(self.targetReachedInfoText)
    self.targetReachedTrueStyleSheet = self.targetReachedStyleSheet + "color: rgb(0,216,25)"
    self.targetReachedYellowAreaStyleSheet = self.targetReachedStyleSheet + "color: rgb(233,236,68)"
    self.targetReachedFalseStyleSheet = self.targetReachedStyleSheet + "color: rgb(255,0,0)"

    ##--------------------------------------------------
    ##------------ STOP SIMULATION ------------##
    ## Stop Simulation Button
    self.stopSimulationRepetitionButton = qt.QPushButton("STOP")
    self.stopSimulationRepetitionButton.setStyleSheet(self.pushButtonStyleSheet)
    self.stopSimulationRepetitionButton.enabled = False
    simulationDRR_GroupBox_Layout.addRow(self.stopSimulationRepetitionButton)

    #---------------------------#
    #--------- Results ---------#
    #---------------------------#
    self.simulationResults_GroupBox = ctk.ctkCollapsibleGroupBox()
    self.simulationResults_GroupBox.setTitle("Results")
    self.simulationResults_GroupBox.enabled = True
    self.simulationResults_GroupBox.collapsed = True
    simulationFormLayout.addRow(self.simulationResults_GroupBox)
    simulationResults_GroupBox_Layout = qt.QFormLayout(self.simulationResults_GroupBox)
    simulationResults_H_Layout = qt.QHBoxLayout()
    simulationResults_GroupBox_Layout.addRow(simulationResults_H_Layout)

    ## Total Time text
    self.totalTime_InfoLabel = qt.QLabel('Repetition Time: ')
    self.totalTime_InfoText = qt.QLabel('-')
    self.logic.totalTime_displayText = self.totalTime_InfoText
    simulationResults_GroupBox_Layout.addRow(self.totalTime_InfoLabel, self.totalTime_InfoText)

    ## Total Number of Projections
    self.totalNumberOfProjections_InfoLabel = qt.QLabel('Number of Projections: ')
    self.totalNumberOfProjections_InfoText = qt.QLabel('-')
    self.logic.totalNumberOfProjections_displayText = self.totalNumberOfProjections_InfoText
    simulationResults_GroupBox_Layout.addRow(self.totalNumberOfProjections_InfoLabel, self.totalNumberOfProjections_InfoText)

    ## Total Time and Projections Stimation Time
    self.estimatedSurgicalTime_InfoLabel = qt.QLabel('Estimated Surgical Time: ')
    self.estimatedSurgicalTime_InfoText = qt.QLabel('-')
    self.logic.estimatedSurgicalTime_displayText = self.estimatedSurgicalTime_InfoText
    simulationResults_GroupBox_Layout.addRow(self.estimatedSurgicalTime_InfoLabel, self.estimatedSurgicalTime_InfoText)

    ## Number of punctures
    self.numberOfPuncturesSpinBox = ctk.ctkDoubleSpinBox()
    self.numberOfPuncturesSpinBox.value = 0
    self.numberOfPuncturesSpinBox.decimals = 0
    self.numberOfPuncturesSpinBox.singleStep = 1
    simulationResults_GroupBox_Layout.addRow("Number of Punctures: ", self.numberOfPuncturesSpinBox)

    newRepetitionOrUserButtons_Layout = qt.QHBoxLayout()
    simulationResults_GroupBox_Layout.addRow(newRepetitionOrUserButtons_Layout)
    ## New Repetition button
    self.newRepetitionButton = qt.QPushButton("NEW REPETITION")
    self.newRepetitionButton.enabled = False
    newRepetitionOrUserButtons_Layout.addWidget(self.newRepetitionButton)

    ################################################################################
    ############################### CONNECTIONS ####################################
    ################################################################################
    ##--------Init--------##
    self.initViewPointButton.connect('clicked(bool)', self.onInitViewPointButtonClicked)
    # connect to plus and load
    self.connectToPlusButton.connect('clicked(bool)', self.onConnectToPlusButtonClicked)
    self.loadDataButton.connect('clicked(bool)', self.onLoadDataButtonClicked)
    # done with init button
    self.doneWithInitButton.connect('clicked(bool)', self.onDoneWithInitButtonClicked)
    ##--------Simulation--------##
    self.simulationViewPointButton.connect('clicked(bool)', self.onSimulationViewPointButtonClicked)
    self.initViewPointButton2.connect('clicked(bool)', self.onInitViewPointButtonClicked)
    # user INFO
    self.modeNone_radioButton.connect('clicked(bool)', self.onTargetSelected)
    self.modeS3L_radioButton.connect('clicked(bool)', self.onTargetSelected)
    self.modeS3R_radioButton.connect('clicked(bool)', self.onTargetSelected)
    self.modeS4L_radioButton.connect('clicked(bool)', self.onTargetSelected)
    self.modeS4R_radioButton.connect('clicked(bool)', self.onTargetSelected)
    # general simulate DRR
    self.startSimulationRepetitionButton.connect('clicked(bool)', self.onStartSimulationRepetitionButtonClicked)
    self.focalPointSpinBox.connect("valueChanged(double)", self.onFocalPointSpinBoxValueChanged)
    self.drrThresholdSpinBox.connect("valueChanged(double)", self.onDRRThresholdpinBoxValueChanged)
    self.drrSizeXSpinBox.connect("valueChanged(double)", self.onDRRSizeXSpinBoxValueChanged)
    self.drrSizeYSpinBox.connect("valueChanged(double)", self.onDRRSizeYSpinBoxValueChanged)
    self.makeLateralProjectionButton.connect('clicked(bool)', self.onMakeLateralProjectionButtonClicked)
    self.makeAnteriorProjectionButton.connect('clicked(bool)', self.onMakeAnteriorProjectionButtonClicked)
    self.DRRAddPunctureButton.connect('clicked(bool)', self.onDRRAddPunctureButtonClicked)
    self.DRRRemovePunctureButton.connect('clicked(bool)', self.onDRRRemovePunctureButtonClicked)
    self.targetReachedButton.connect('clicked(bool)', self.onTargetReachedButtonClicked)
    self.stopSimulationRepetitionButton.connect('clicked(bool)', self.onStopSimulationRepetitionButtonClicked)
    # results
    self.numberOfPuncturesSpinBox.connect("valueChanged(double)", self.onNumberOfPuncturesSpinBoxValueChanged)
    self.newRepetitionButton.connect('clicked(bool)', self.onNewRepetitionButtonClicked)


    # Add vertical spacer
    self.layout.addStretch(1)

  #----------------------------------------------------
  # Init
  #----------------------------------------------------
  def onInitViewPointButtonClicked(self):
    self.layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    self.simulationViewPointButton.enabled = True
    self.initViewPointButton.enabled = False
    self.initViewPointButton2.enabled = False

  def onSimulationViewPointButtonClicked(self):
    self.layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)
    self.simulationViewPointButton.enabled = False
    self.initViewPointButton.enabled = True
    self.initViewPointButton2.enabled = True

  def loadSelectedPhantom(self, phantomID):
    self.phantomID = "Phantom00"
    # Record activity
    self.logic.recordSoftwareActivity('Phantom Selected')

    self.logic.selectPhantomID(phantomID)

    # Update button state
    self.connectToPlusButton.enabled = True
    self.phantomSelection_GroupBox.collapsed = True
    self.connections_GroupBox.collapsed = False

  def onConnectToPlusButtonClicked(self):

    # Record activity
    self.logic.recordSoftwareActivity('Connect To Plus')

    # Update button state
    self.connectToPlusButton.enabled = False

    # Update connection
    if self.connect:
      port_tracker = 18944
      status = self.logic.startPlusConnection(port_tracker)  # Start connection
      if status == 1:
        self.connect = False
        self.connectToPlusButton.setText('Disconnect from Plus')
    else:
      self.logic.stopPlusConnection()  # Stop connection
      self.connect = True
      self.connectToPlusButton.setText('Connect to Plus')

    # Update button state
    self.connectToPlusButton.enabled = True

  def onLoadDataButtonClicked(self):
    self.phantomID = "Phantom01"
    self.logic.selectPhantomID(self.phantomID)

    # Record activity
    self.logic.recordSoftwareActivity('Load Data')

    # Update button state
    self.loadDataButton.enabled = False

    # Load data
    self.logic.loadData()

    # Build transformation tree
    self.logic.buildTransformationTree()

    # Update button state
    self.connections_GroupBox.collapsed = True

    # Update viewpoint
    # self.logic.updateViewpoint(cameraID='FRONT')
    # self.logic.updateDualViewSelection(side=1, selection=3)

  def onDoneWithInitButtonClicked(self):

    self.initCollapsibleButton.collapsed = True
    self.simulationCollapsibleButton.collapsed = False
    self.simulationViewPointButton.enabled = False
    self.initViewPointButton.enabled = True
    self.initViewPointButton2.enabled = True
    self.layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)

    self.modeSelected = "1"
    self.logic.modeSelected = "1"
    self.makeProjectionMode1_GroupBox.visible = True

  #----------------------------------------------------
  # SIMULATION
  #----------------------------------------------------

  def onTargetSelected(self):
    targetName = "None"

    if self.modeNone_radioButton.isChecked():
      self.startSimulationRepetitionButton.enabled = False
      targetName = "None"

    if self.modeS3L_radioButton.isChecked():
      self.startSimulationRepetitionButton.enabled = True
      targetName = "S3L"

    if self.modeS3R_radioButton.isChecked():
      self.startSimulationRepetitionButton.enabled = True
      targetName = "S3R"

    if self.modeS4L_radioButton.isChecked():
      self.startSimulationRepetitionButton.enabled = True
      targetName = "S4L"

    if self.modeS4R_radioButton.isChecked():
      self.startSimulationRepetitionButton.enabled = True
      targetName = "S4R"

    self.targetSelected = targetName

  def onStartSimulationRepetitionButtonClicked(self):
    ## start new logger:
    log_name = "TraditionalMethod_User{}_Rep{}_Target{}".format(self.userID, self.repetitionID, self.targetSelected)
    self.rep_log_path = r"C:\Projects\tmp\LOG_SlicerModule_{}_{}.log".format(log_name, time.strftime("%Y-%m-%d_%H-%M-%S"))
    self.rep_log = MyLog(log_file_path=self.rep_log_path, log_name=log_name)
    self.rep_log.init_log()
    self.logic.rep_log = self.rep_log

    self.rep_log.log("[START-SIMU] Simulation started at: {}".format(time.strftime("%H:%M:%S", time.localtime())))

    ## init var
    self.repetitionStartTime = time.time()
    self.repetitionNumberOfProjections = 0
    self.repetitionComputationalTotalTime = 0
    self.timesTargetReachedButtonClicked = 0
    self.singleProjectionStartTime = self.repetitionStartTime

    self.logic.startSimulationRepetition(self.targetSelected)

    ## Update layout
    self.startSimulationRepetitionButton.enabled = False
    self.stopSimulationRepetitionButton.enabled = True
    self.DRRAddPunctureButton.enabled = True
    self.DRRRemovePunctureButton.enabled = True
    self.userInformation_GroupBox.collapsed = True
    self.targetReachedButton.enabled = True

    self.makeProjectionMode1_GroupBox.enabled = True

    pass

  def onFocalPointSpinBoxValueChanged(self, value):
    self.rep_log.log("[FOCALPOINT] Focal point change...")
    self.logic.focalPoint = value

  def onDRRThresholdpinBoxValueChanged(self, value):
    self.logic.drrThreshold = value

  def onDRRSizeXSpinBoxValueChanged(self, value):
    self.logic.drrSizeX = value

  def onDRRSizeYSpinBoxValueChanged(self, value):
    self.logic.drrSizeY = value

  def onMakeLateralProjectionButtonClicked(self):
    self.rep_log.log("[LATERALPROJ] Lateral projection button clicked.")
    self.onMakeProjectionButtonClicked("mode1_lateral")

  def onMakeAnteriorProjectionButtonClicked(self):
    self.rep_log.log("[ANTERIORPROJ] Anterior projection button clicked.")
    self.onMakeProjectionButtonClicked("mode1_anterior")

  def onMakeProjectionButtonClicked(self, projectionType):
    self.rep_log.log("[onMAKE-PROJ] Make Projection Clicked.")

    # update variables
    aux = self.singleProjectionStartTime
    self.singleProjectionStartTime = time.time()
    self.logic.updateDATA("TimeAtEachProjection", self.singleProjectionStartTime - self.repetitionStartTime)
    self.logic.updateDATA("TimePerProjection", self.singleProjectionStartTime - aux)
    self.repetitionNumberOfProjections += 1

    # Make action
    self.logic.makeProjection(projectionType=projectionType)

    # update variables
    self.singleProjectionComputationalTime = time.time() - self.singleProjectionStartTime
    self.repetitionComputationalTotalTime += self.singleProjectionComputationalTime
    self.logic.updateDATA("ComputationalTimePerProjection", self.singleProjectionComputationalTime)

    # update layout
    self.simulationViewPointButton.enabled = False
    self.initViewPointButton.enabled = True
    self.initViewPointButton2.enabled = True

  def onDRRAddPunctureButtonClicked(self):
    self.numberOfPunctures += 1
    self.DRRNumberOfPunctures_InfoText.setText("Number Of Punctures: {}".format(self.numberOfPunctures))
    self.rep_log.log("[ADDPUNCTURE] Puncture added: {}".format(self.numberOfPunctures))

  def onDRRRemovePunctureButtonClicked(self):
    if self.numberOfPunctures != 0:
      self.numberOfPunctures -= 1

    self.DRRNumberOfPunctures_InfoText.setText("Number Of Punctures: {}".format(self.numberOfPunctures))
    self.rep_log.log("[REMOVEPUNCTURE] Puncture removed: {}".format(self.numberOfPunctures))

  def onTargetReachedButtonClicked(self):
    timeButtonClicked = time.time() - self.repetitionStartTime
    isTargetReached = self.logic.isNeedleTipInTargetArea()
    self.timesTargetReachedButtonClicked += 1

    if isTargetReached == "GreenArea":
      self.targetReachedInfoText.setText("YES!!!")
      self.targetReachedInfoText.setStyleSheet(self.targetReachedTrueStyleSheet)
    elif isTargetReached == "YellowArea":
      self.targetReachedInfoText.setText("YOU ARE CLOSE!!!")
      self.targetReachedInfoText.setStyleSheet(self.targetReachedYellowAreaStyleSheet)
    else:
      self.targetReachedInfoText.setText("NO, Try Again!!!")
      self.targetReachedInfoText.setStyleSheet(self.targetReachedFalseStyleSheet)

    self.rep_log.log("[TARGETREACHED] Is Needle In Target button CLICKED. RESULT: {}".format(isTargetReached))

    ## update results
    self.logic.updateDATA("OutputPerTargetReachedButtonClicked", isTargetReached)
    self.logic.updateDATA("TimeAtEachTargetReachedButtonClicked", timeButtonClicked)

  def onStopSimulationRepetitionButtonClicked(self):
    self.repetitionStopTime = time.time()
    self.repetitionTotalTime = self.repetitionStopTime - self.repetitionStartTime
    self.logic.updateDATA("TimePerProjection", self.repetitionStopTime - self.singleProjectionStartTime)
    self.logic.updateDATA("NumberOfTimesTargetReachedButtonClicked", self.timesTargetReachedButtonClicked)

    self.rep_log.log("[STOP] Repetition stopped: {}".format(self.repetitionTotalTime))

    ## Update layout
    # self.startSimulationRepetitionButton.enabled = False
    if self.modeSelected == "1":
      self.makeProjectionMode1_GroupBox.enabled = False
    self.stopSimulationRepetitionButton.enabled = False
    self.DRRAddPunctureButton.enabled = False
    self.DRRRemovePunctureButton.enabled = False
    self.simulationDRR_GroupBox.collapsed = True
    self.simulationResults_GroupBox.collapsed = False
    self.targetReachedButton.enabled = False
    self.numberOfPuncturesSpinBox.value = self.numberOfPunctures
    self.newRepetitionButton.enabled = True

    ## update Results
    self.updateResultsInfo()

  #----------------------------------------------------
  # RESULTS AND SAVE
  #----------------------------------------------------
  def updateResultsInfo(self):
    ## Estmiate Surgical Time
    repetitionEstimatedSurgicalTime = self.logic.calculateSurgicalTimeFromProjections(self.repetitionNumberOfProjections,
                                                                                self.repetitionComputationalTotalTime,
                                                                                self.repetitionTotalTime)

    self.totalTime_InfoText.setText(time.strftime('%M:%S', time.gmtime(self.repetitionTotalTime)))
    self.totalNumberOfProjections_InfoText.setText(str(self.repetitionNumberOfProjections))
    self.estimatedSurgicalTime_InfoText.setText(time.strftime('%M:%S', time.gmtime(repetitionEstimatedSurgicalTime)))

    ## Update DATA DICT
    self.logic.updateDATA("NumberOfProjections", self.repetitionNumberOfProjections)
    self.logic.updateDATA("RepetitionTotalTime", self.repetitionTotalTime)
    self.logic.updateDATA("EstimatedSurgicalTime", repetitionEstimatedSurgicalTime)
    self.logic.updateDATA("TargetSelected", self.targetSelected)

    pass

  def onNumberOfPuncturesSpinBoxValueChanged(self, value):
    self.numberOfPunctures = value
    self.rep_log.log("Number of punctures: {}".format(value))

  def onNewRepetitionButtonClicked(self):
    try:
      self.repetitionID = int(self.repetitionID) + 1
    except:
      pass

    # update layout
    self.simulationResults_GroupBox.collapsed = True
    self.simulationDRR_GroupBox.collapsed = False
    self.userInformation_GroupBox.collapsed = False
    self.startSimulationRepetitionButton.enabled = False
    self.newRepetitionButton.enabled = False

    self.resetVariablesAndTexts()

    self.rep_log.log("[NEWREP] New repetition button clciked.")

  def resetVariablesAndTexts(self):
    self.totalTime_InfoText.setText("-")
    self.estimatedSurgicalTime_InfoText.setText("-")
    self.totalNumberOfProjections_InfoText.setText("-")
    self.targetReachedInfoText.setText("-")
    self.numberOfPunctures = 0
    self.timesTargetReachedButtonClicked = 0
    self.DRRNumberOfPunctures_InfoText.setText("Number Of Punctures: {}".format(self.numberOfPunctures))

    self.modeNone_radioButton.checked = True
    self.targetSelected = "None"

    self.logic.resetSimulationLayout()

  def setCustomLayout(self):
    layoutLogic = self.layoutManager.layoutLogic()
    customLayout_1 = ("<layout type=\"horizontal\">"
                      " <item>"
                      "  <view class=\"vtkMRMLViewNode\" singletontag=\"1\">"
                      "   <property name=\"viewlabel\" action=\"default\">1</property>"
                      "  </view>"
                      " </item>"
                      "</layout>")
    customLayout_2 = ("<layout type=\"horizontal\" split=\"true\">"
                      " <item>"
                      "  <view class=\"vtkMRMLSliceNode\" singletontag=\"Yellow\">"
                      "   <property name=\"orientation\" action=\"default\">Sagittal</property>"
                      "   <property name=\"viewlabel\" action=\"default\">Y</property>"
                      "   <property name=\"viewcolor\" action=\"default\">#F34A33</property>"
                      "  </view>"
                      " </item>"
                      " <item>"
                      "  <view class=\"vtkMRMLViewNode\" singletontag=\"1\">"
                      "  <property name=\"viewlabel\" action=\"default\">T</property>"
                      "  </view>"
                      " </item>"
                      "</layout>")
    customLayout_3 = ("<layout type=\"horizontal\">"
                      " <item>"
                      "  <view class=\"vtkMRMLSliceNode\" singletontag=\"Red\">"
                      "   <property name=\"orientation\" action=\"default\">Axial</property>"
                      "     <property name=\"viewlabel\" action=\"default\">R</property>"
                      "     <property name=\"viewcolor\" action=\"default\">#F34A33</property>"
                      "  </view>"
                      " </item>"
                      "</layout>")
    customLayout_4 = ("<layout type=\"horizontal\" split=\"false\">"
                      " <item>"
                      "  <view class=\"vtkMRMLViewNode\" singletontag=\"1\">"
                      "  <property name=\"viewlabel\" action=\"default\">1</property>"
                      "  </view>"
                      " </item>"
                      " <item>"
                      "  <view class=\"vtkMRMLViewNode\" singletontag=\"2\">"
                      "  <property name=\"viewlabel\" action=\"default\">2</property>"
                      "  </view>"
                      " </item>"
                      "</layout>")
    customLayout_5 = ("<layout type=\"vertical\">"
                      " <item>"
                      "  <layout type=\"horizontal\">"
                      "   <item>"
                      "    <view class=\"vtkMRMLViewNode\" singletontag=\"1\">"
                      "     <property name=\"viewlabel\" action=\"default\">1</property>"
                      "    </view>"
                      "   </item>"
                      "   <item>"
                      "    <view class=\"vtkMRMLViewNode\" singletontag=\"2\">"
                      "     <property name=\"viewlabel\" action=\"default\">2</property>"
                      "    </view>"
                      "   </item>"
                      "  </layout>"
                      " </item>"
                      " <item>"
                      "  <layout type=\"horizontal\">"
                      "   <item>"
                      "    <view class=\"vtkMRMLViewNode\" singletontag=\"3\">"
                      "     <property name=\"viewlabel\" action=\"default\">3</property>"
                      "    </view>"
                      "   </item>"
                      "   <item>"
                      "    <view class=\"vtkMRMLViewNode\" singletontag=\"4\">"
                      "     <property name=\"viewlabel\" action=\"default\">4</property>"
                      "    </view>"
                      "   </item>"
                      "  </layout>"
                      " </item>"
                      "</layout>")
    self.customLayout_1_ID = 996
    self.customLayout_2_ID = 997
    self.customLayout_3_ID = 998
    self.customLayout_4_ID = 999
    self.customLayout_5_ID = 1000
    layoutLogic.GetLayoutNode().AddLayoutDescription(self.customLayout_1_ID, customLayout_1)
    layoutLogic.GetLayoutNode().AddLayoutDescription(self.customLayout_2_ID, customLayout_2)
    layoutLogic.GetLayoutNode().AddLayoutDescription(self.customLayout_3_ID, customLayout_3)
    layoutLogic.GetLayoutNode().AddLayoutDescription(self.customLayout_4_ID, customLayout_4)
    layoutLogic.GetLayoutNode().AddLayoutDescription(self.customLayout_5_ID, customLayout_5)


#
# SNSClinicalSimulationLogic
#

class SNSClinicalSimulationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):

    ## Paths
    self.main_resources_path = slicer.modules.snsclinicalsimulation.path.replace(r"SNSClinicalSimulation/SNSClinicalSimulation.py", "") + "Resources"
    self.module_path = slicer.modules.snsclinicalsimulation.path.replace("SNSClinicalSimulation.py", "") + "Resources"
    self.phantomData_path = os.path.join(self.main_resources_path, "PhantomsData")
    self.data_path = os.path.join(self.main_resources_path, "Data")
    self.models_path = os.path.join(self.main_resources_path, "Models")
    self.module_results_path = os.path.join(self.main_resources_path, "Results")

    ## Init
    self.utils = Utils()

    ## Transformation
    self.StylusTipToStylus = None
    self.StylusToTracker = None
    self.TrackerToReference = None
    self.ReferenceToRAS = None
    self.NeedleTipToNeedle = None
    self.NeedleToTracker = None

    ## Init Variables
    self.phantomID = None
    self.rep_log = None

    self.RF_registration_RMS = 0
    self.RF_fiducials_displayText = None

    self.modeSelected = "1"

    self.existingTransformsMatrixDICT = {}

    # Registration error
    self.RF_registration_RMS = 0
    self.RF_registration_RMS_displayText = None

    # Pivot Calibration
    self.pivotCalibrationLogic = slicer.modules.pivotcalibration.logic()

    # DRR params
    self.focalPoint = 400
    self.drrThreshold = 100
    self.drrSizeX = 512
    self.drrSizeY = 512

    self.DRR1ProjArray = None
    self.DRR2ProjArray = None

    # LayoutManager
    self.layoutManager = slicer.app.layoutManager()
    self.red_logic = self.layoutManager.sliceWidget("Red").sliceLogic()
    self.yellow_logic = self.layoutManager.sliceWidget("Yellow").sliceLogic()

    # breach warning
    self.breachWarningLogic = slicer.modules.breachwarning.logic()

    # Watchdog
    wd_logic = slicer.vtkSlicerWatchdogLogic()
    wd_logic.AddNewWatchdogNode('WatchdogNode', slicer.mrmlScene)
    self.wd = slicer.util.getNode('WatchdogNode')
    self.wd.GetDisplayNode().SetOpacity(0.0)  # Make background transparent
    self.wd.GetDisplayNode().SetColor(0.0,0.0,0.0)  # Set color of watchdog message text to black
    self.wd.GetDisplayNode().SetFontSize(25)  # Set watchdog font size

    # Activity recording
    self.recordedActivity_action = list()
    self.recordedActivity_timeStamp = list()


  def selectPhantomID(self, phantomID):
    self.phantomID = phantomID
    self.phantomData_path = os.path.join(self.phantomData_path, phantomID)

    print("Phantom Selected: ", self.phantomID)

  def startPlusConnection(self, port_tracker):

    # Open connection
    try:
      cnode = slicer.util.getNode('IGTLConnector_Tracker')
    except:
      cnode = slicer.vtkMRMLIGTLConnectorNode()
      slicer.mrmlScene.AddNode(cnode)
      cnode.SetName('IGTLConnector_Tracker')
    status = cnode.SetTypeClient('localhost', port_tracker)

    # Check connection status
    if status == 1:
      cnode.Start()
      logging.debug('Connection Successful')

    else:
      print('ERROR: Unable to connect to PLUS')
      logging.debug('ERROR: Unable to connect to PLUS')

    return status

  def stopPlusConnection(self):

    cnode = slicer.util.getNode('IGTLConnector_Tracker')
    cnode.Stop()

  def loadData(self):
    print("[LOADDATA] Loading Data...")
    ## Load Transfroms
    self.StylusTipToStylus = self.utils.loadTransformFromFile('StylusTipToStylus', os.path.join(self.data_path, "StylusTipToStylus.h5"))
    self.StylusToTracker = self.utils.getOrCreateTransform('StylusToTracker')
    self.TrackerToReference = self.utils.getOrCreateTransform('TrackerToReference')
    self.NeedleToTracker = self.utils.getOrCreateTransform('NeedleToTracker')

    self.ReferenceToRAS = self.utils.loadTransformFromFile('ReferenceToRAS', os.path.join(self.data_path, "ReferenceToRAS.h5"))
    self.updateOrLoadExistingTransform(self.ReferenceToRAS)
    self.makeTransformIdentity(self.ReferenceToRAS)

    self.NeedleTipToNeedle = self.utils.loadTransformFromFile('NeedleTipToNeedle', os.path.join(self.data_path, "NeedleTipToNeedle.h5"))
    self.updateOrLoadExistingTransform(self.NeedleTipToNeedle)
    self.makeTransformIdentity(self.NeedleTipToNeedle)

    ## Load Generic Models
    self.stylusModelNode = self.utils.loadModelFromFile("StylusModel", os.path.join(self.models_path, "StylusModel.stl"), color=[0,0,0])
    self.needleModelNode = self.utils.loadModelFromFile("NeedleModel", os.path.join(self.models_path, "SacralNeedleModel.stl"), color=[1,0,0])

    ## Load Phantom Models
    self.boneModelNode = self.utils.loadModelFromFile("Bone", os.path.join(self.phantomData_path, "Bone.stl"), color=[1,1,1])
    self.skinModelNode = self.utils.loadModelFromFile("Skin", os.path.join(self.phantomData_path, "SoftTissue.stl"), color=[241/255,214/255,145/255])

    ## Load Volume ##
    self.phantomVolumeNode = self.utils.loadVolumeFromFile("PhantomCT",  os.path.join(self.phantomData_path, "PhantomCT.nrrd"))
    self.phantomVolumeArray = self.getVolumeArrayFromVolumeNode(self.phantomVolumeNode)
    # self.phantomVolumeNode.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
    self.DRR1VolumeNode = self.utils.getOrCreateVolume("DRR1")
    slicer.util.updateVolumeFromArray(self.DRR1VolumeNode, np.zeros((1, 512, 512), dtype="int16"))
    self.red_logic.GetSliceCompositeNode().SetBackgroundVolumeID(self.DRR1VolumeNode.GetID())
    self.DRR2VolumeNode = self.utils.getOrCreateVolume("DRR2")  # color_table="vtkMRMLColorTableNodeInvertedGrey"
    slicer.util.updateVolumeFromArray(self.DRR2VolumeNode, np.zeros((1, 512, 512), dtype="int16"))
    self.yellow_logic.GetSliceCompositeNode().SetBackgroundVolumeID(self.DRR2VolumeNode.GetID())
    self.layoutManager.sliceWidget("Yellow").mrmlSliceNode().SetOrientationToAxial()

    # Center 3D view
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()

    print("[LOADDATA] Data loaded.")

  def updateOrLoadExistingTransform(self, transformNode):
    transformName = transformNode.GetName()

    keys = self.existingTransformsMatrixDICT.keys()

    if transformName not in keys:
      m = vtk.vtkMatrix4x4()
      transformNode.GetMatrixTransformToWorld(m)
      self.existingTransformsMatrixDICT[transformName] = m
    else:
      transformNode.SetMatrixTransformToParent(self.existingTransformsMatrixDICT[transformName])

  def buildTransformationTree(self):
    print("[TR-TREE] Building Transform Tree...")
    # Build transform tree
    self.stylusModelNode.SetAndObserveTransformNodeID(self.StylusTipToStylus.GetID())
    self.needleModelNode.SetAndObserveTransformNodeID(self.NeedleTipToNeedle.GetID())

    self.StylusTipToStylus.SetAndObserveTransformNodeID(self.StylusToTracker.GetID())
    self.NeedleTipToNeedle.SetAndObserveTransformNodeID(self.NeedleToTracker.GetID())

    self.StylusToTracker.SetAndObserveTransformNodeID(self.TrackerToReference.GetID())
    self.NeedleToTracker.SetAndObserveTransformNodeID(self.TrackerToReference.GetID())

    self.TrackerToReference.SetAndObserveTransformNodeID(self.ReferenceToRAS.GetID())

    # Add watchdogs
    self.removeAllWatchedNodes()
    self.TrackerToStylus = self.utils.getOrCreateTransform("TrackerToStylus")
    self.TrackerToNeedle = self.utils.getOrCreateTransform("TrackerToNeedle")
    self.addWatchdog(self.TrackerToStylus, watchedNodeID=0, warningMessage='Stylus is out of view', playSound=True)
    self.addWatchdog(self.TrackerToReference, watchedNodeID=1, warningMessage='Reference is out of view',
                     playSound=True)
    self.addWatchdog(self.TrackerToNeedle, watchedNodeID=2, warningMessage='Needle is out of view',
                     playSound=True)

    print("[TR-TREE] Transform tree Built.")

  #----------------------------------------------------
  # DRR Projection
  #----------------------------------------------------
  def makeProjection(self, projectionType=None):

    ## 1. Create segmentation from model
    needleModelHardenNode, needlePositionTransform = self.copyAndHardenModel(self.needleModelNode)
    [success, self.segmentationNode] = self.createSegmentationFromModel(needleModelHardenNode, self.phantomVolumeNode)
    self.segmentationNode.GetDisplayNode().SetVisibility(0)
    # slicer.mrmlScene.RemoveNode(needleModelHardenNode)

    matrixArray = self.utils.getMatrixArrayFromTransformNode(needlePositionTransform)
    self.updateDATA("NeedlePositionTransforms", matrixArray)


    ## 2. Create LabelMap from segmentation
    [success, self.labelMapNode] = self.createLabelMapVolumeFromSegmentation(self.segmentationNode, self.phantomVolumeNode)
    labelMapArray = self.getVolumeArrayFromVolumeNode(self.labelMapNode)

    ## 3. Update CT with LabelMap and Value
    ctValue = 1500
    DRRCTVolumeArray = self.setCTValueToModel(self.phantomVolumeArray, labelMapArray, ctValue)
    slicer.util.updateVolumeFromArray(self.phantomVolumeNode, DRRCTVolumeArray)

    ## 4. Get params for projection
    DRRParams = self.getDRRParams(projectionType)

    ## 5. Make projection
    if projectionType=="mode1_lateral":
      DRRVolumeNode = self.DRR1VolumeNode
      self.DRR1ProjArray = True
    elif projectionType=="mode1_anterior":
      DRRVolumeNode = self.DRR2VolumeNode
      self.DRR2ProjArray = True
    else:
      DRRVolumeNode = self.DRR1VolumeNode

    projArray = self.generateDRR(self.phantomVolumeNode, DRRVolumeNode, DRRParams)
    self.updateDATA("Projections", projArray)

    ## 4. Update Slicer view
    self.updateSimulationLayout(DRR1=self.DRR1ProjArray, DRR2=self.DRR2ProjArray)

  def getDRRParams(self, projectionType):
    DRRParamsMatrixArray = None

    if self.modeSelected == "1":
      if projectionType == "mode1_anterior":
        # DRRParams["axis"] = 1
        t = [10, 85, 0]
        r = [90, 0, 180]

      elif projectionType == "mode1_lateral":
        # DRRParams["axis"] = 2
        t = [75, 50, 0]
        r = [180, -90, 0]

      else:
        t = [0, 100, 0]
        r = [90, 180, 0]

      DRRParamsMatrix = self.utils.setTranslationAndRotationToVTK(t[0], t[1], t[2], r[0], r[1], r[2])
      DRRParamsMatrixArray = self.utils.getMatrixArrayFromVTKMatrix(DRRParamsMatrix.GetMatrix())

    elif self.modeSelected == "2":
      if projectionType == "mode2_RBParams":
        ## Get Origin To Volume Center
        bounds = np.zeros(6)
        self.phantomVolumeNode.GetBounds(bounds)
        centerVolume = np.array([(bounds[1] + bounds[0]) / 2,
                                 (bounds[3] + bounds[2]) / 2,
                                 (bounds[5] + bounds[4]) / 2])
        OriginToVolCenterMatrix = vtk.vtkTransform()
        OriginToVolCenterMatrix.Translate(centerVolume[0], centerVolume[1], centerVolume[2])
        C = self.utils.getMatrixArrayFromVTKMatrix(OriginToVolCenterMatrix.GetMatrix())

        ## Get Focal Point To DRR Transform matrix
        FocalPointToDRRTransformMatrix = self.utils.setTranslationAndRotationToVTK(0, 0, -self.focalPoint/2,
                                                                                   180, 0, 0)
        F = self.utils.getMatrixArrayFromVTKMatrix(FocalPointToDRRTransformMatrix.GetMatrix())

        ##
        XRayTubeToProjectionPosition = self.utils.getOrCreateTransform("XRayTubeToProjectionPosition")
        self.makeTransformIdentity(XRayTubeToProjectionPosition)
        XRayTubeToProjectionPosition.SetAndObserveTransformNodeID(self.XRaytubeToXRaytubeReference.GetID())
        XRayTubeToProjectionPosition.HardenTransform()
        T = self.utils.getMatrixArrayFromTransformNode(XRayTubeToProjectionPosition)

        # ## FINAL TRANSFORM
        C_inv = np.linalg.inv(C)
        F_inv = np.linalg.inv(F)
        DRRParamsMatrixArray = np.linalg.multi_dot([C_inv, T, F_inv])

    DRRParams = self.setDRRParams(DRRParamsMatrix=DRRParamsMatrixArray, sid=self.focalPoint,
                                  drrthreshold=self.drrThreshold, drrsizex=self.drrSizeX, drrsizey=self.drrSizeY)

    return DRRParams

  def updateSimulationLayout(self, DRR1=False, DRR2=False):
    self.layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)

    self.red_logic.GetSliceCompositeNode().SetBackgroundVolumeID(self.DRR1VolumeNode.GetID())
    self.yellow_logic.GetSliceCompositeNode().SetBackgroundVolumeID(self.DRR2VolumeNode.GetID())

    self.layoutManager.sliceWidget("Yellow").mrmlSliceNode().SetOrientationToAxial()

    if DRR1:
      self.DRR1VolumeNode.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeInvertedGrey")
    if DRR2:
      self.DRR2VolumeNode.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeInvertedGrey")

    slicer.util.resetSliceViews()

    # for sliceViewName in self.layoutManager.sliceViewNames():
    #   print("Reseting View {}".format(sliceViewName))
    #   self.layoutManager.sliceWidget(sliceViewName).mrmlSliceNode().SetOrientationToSagittal()

  def resetSimulationLayout(self):
    slicer.util.updateVolumeFromArray(self.DRR1VolumeNode, np.zeros((1, 512, 512), dtype="int16"))
    slicer.util.updateVolumeFromArray(self.DRR2VolumeNode, np.zeros((1, 512, 512), dtype="int16"))

    self.updateSimulationLayout()

    return

  def createSegmentationFromModel(self, modelNode, volumeNode):
    self.rep_log.log("[SEGMENTATION] Creating segmentation from model...")
    ## Create Segmentation
    segmentationNode = slicer.vtkMRMLSegmentationNode()
    segmentationNode.SetName("SegmentationModel")

    ## Assign volume to Segmentation
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    slicer.mrmlScene.AddNode(segmentationNode)

    ## Create Segmentation segment from model (importToCurrentSegmentation)
    success = slicer.vtkSlicerSegmentationsModuleLogic().ImportModelToSegmentationNode(modelNode, segmentationNode)
    self.rep_log.log("[SEGMENTATION] DONE.")
    return success, segmentationNode

  def createLabelMapVolumeFromSegmentation(self, segmentationNode, volumeNode):
    self.rep_log.log("[LABELMAP] Creating labelmap from segmentation of the model...")

    # segmentID = self.segmentationNode.GetSegmentation().GetSegment()

    ## Create LabelMap Node
    labelmapNode = self.utils.createLabelMapVolumeNode("LabelMapModel")

    success = slicer.vtkSlicerSegmentationsModuleLogic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelmapNode, volumeNode)

    self.rep_log.log("[LABELMAP] DONE")
    return success, labelmapNode

  def setCTValueToModel(self, volume_array, labelmap_array, ctValue):
    volume_model_array = np.copy(volume_array)
    idx_model = np.where(labelmap_array != 0)
    volume_model_array[idx_model] = ctValue

    return volume_model_array

  def fromVolumeNodeToITKImage(self, volumeNode):

    ######################################
    # Get Metainfo from volume
    ######################################
    metainfo = {}

    metainfo['spacing'] = volumeNode.GetSpacing()
    metainfo['space origin'] = volumeNode.GetOrigin()
    metainfo['size'] = volumeNode.GetImageData().GetDimensions()
    # metainfo['bounds'] = np.zeros(6)
    # volume.GetBounds(metainfo['bounds'])
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASDirectionMatrix(matrix)
    matrix = self.utils.ArrayFromVTKMatrix(matrix)[:3, :3]
    metainfo['space directions'] = itk.GetMatrixFromArray(matrix)
    metainfo['size'] = volumeNode.GetImageData().GetDimensions()

    ######################################
    # More steps
    ######################################
    ## Get numpy from volume
    vol_array = slicer.util.arrayFromVolume(volumeNode)
    self.rep_log.log("Volume Array Shape = ", vol_array.shape)

    ## Go From Numpy to ITK Image
    image = itk.image_from_array(vol_array)

    ## Import metadata from volume to itk
    image.SetSpacing(metainfo['spacing'])
    image.SetOrigin(metainfo['space origin'])
    image.SetDirection(metainfo['space directions'])
    # image.SetBufferedRegion(metainfo['image region'])

    return image

  def generateDRR(self, inputVolumeNode, outputVolumeNode, DRRParams):
    ## Set Params
    translation, rot = DRRParams["translation"], DRRParams["rot"]
    cx, cy, cz = DRRParams["center"][0], DRRParams["center"][1], DRRParams["center"][2]
    drrthreshold, sid = DRRParams["drrthreshold"], DRRParams["sid"]
    drrsizex, drrsizey = DRRParams["drrsizex"], DRRParams["drrsizey"]

    image = self.fromVolumeNodeToITKImage(inputVolumeNode)
    image_type = itk.Image[itk.SS, 3]
    output_image_type = itk.Image[itk.SS, 2]

    ######################################
    # Set Transform from user variables
    ######################################
    transform_type = itk.CenteredEuler3DTransform[itk.D]
    transform = transform_type.New()
    transform.SetComputeZYX(True)

    # Translation
    transform.SetTranslation(translation)

    # Rotation
    dtr = np.arctan(1.0) * 4 / 180  ## formula de grados a radianes
    transform.SetRotation(dtr * rot[0], dtr * rot[1], dtr * rot[2])

    # Center
    imOrigin = np.array(image.GetOrigin())
    imRes = image.GetSpacing()
    imRegion = image.GetBufferedRegion()
    imSize = imRegion.GetSize()

    imOrigin[0] += imRes[0] * imSize[0] / 2.0
    imOrigin[1] += imRes[1] * imSize[1] / 2.0
    imOrigin[2] += imRes[2] * imSize[2] / 2.0

    center = np.zeros(3)
    center[0] = cx + imOrigin[0]
    center[1] = cy + imOrigin[1]
    center[2] = cz + imOrigin[2]
    transform.SetCenter(center)

    self.rep_log.log("[DRR] Transform Completed")

    ######################################
    # Set Interpolator
    ######################################
    ray_caster_type = itk.RayCastInterpolateImageFunction[image_type, itk.D]  ## Que formula aplica???????
    interpolator = ray_caster_type.New()
    interpolator.SetTransform(transform)
    interpolator.SetThreshold(drrthreshold)

    # Focal point
    focalpoint = np.zeros(3)
    focalpoint[0] = imOrigin[0]
    focalpoint[1] = imOrigin[1]
    focalpoint[2] = imOrigin[2] - sid / 2
    interpolator.SetFocalPoint(focalpoint)

    self.rep_log.log("[DRR] Interpolator Completed")

    ######################################
    # Set Final Volume Params
    ######################################
    size = np.zeros(3)
    imSize[0] = drrsizex
    imSize[1] = drrsizey
    imSize[2] = 1

    spacing = np.zeros(3)
    spacing[0] = 1
    spacing[1] = 1
    spacing[2] = 1

    origin = np.zeros(3)
    origin[0] = imOrigin[0] + 0 - 1. * (drrsizex - 1.) / 2.
    origin[1] = imOrigin[1] + 0 - 1. * (drrsizey - 1.) / 2.
    origin[2] = imOrigin[2] + sid / 2.;

    self.rep_log.log("[DRR] Final Volume Parameters Completed")

    ######################################
    # Final Fiilter: Resample
    ######################################
    resample_filter = itk.ResampleImageFilter[image_type, image_type].New()
    resample_filter.SetInput(image)
    resample_filter.SetDefaultPixelValue(0)

    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetSize(imSize)
    resample_filter.SetOutputSpacing(spacing)
    resample_filter.SetOutputOrigin(origin)

    self.rep_log.log("[DRR] Resample Filter Done")

    ######################################
    # Rescale Image Values
    ######################################
    rescale_type = itk.RescaleIntensityImageFilter[image_type, image_type]
    rescale_filter = rescale_type.New()
    rescale_filter.SetOutputMinimum(0)
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.SetInput(resample_filter.GetOutput())
    rescale_filter.Update()

    self.rep_log.log("[DRR] Rescale Image Done")

    ######################################
    # Save as NRRD
    ######################################
    output_path = "output_fixed.nrrd"
    # writer = itk.ImageFileWriter.New(rescale_filter.GetOutput(), FileName=output_path)
    # writer.SetFileName(output_path)
    # writer.SetInput(rescale_filter.GetOutput())
    # writer.Update()
    itk.imwrite(rescale_filter.GetOutput(), output_path)

    self.rep_log.log("[DRR] Save as NRRD (first file)")

    ######################################
    # Extract Image filter as another size
    ######################################
    slice3 = itk.ExtractImageFilter[image_type, output_image_type].New()
    slice3.InPlaceOn()
    slice3.SetDirectionCollapseToSubmatrix()
    inputRegion3 = rescale_filter.GetOutput().GetLargestPossibleRegion()
    size3 = inputRegion3.GetSize()
    size3[2] = 0
    start3 = inputRegion3.GetIndex()
    sliceNumber3 = 0  ## cont unsigned integer
    start3[2] = sliceNumber3

    desiredRegion3 = imRegion
    desiredRegion3.SetSize(size3)
    desiredRegion3.SetIndex(start3)
    slice3.SetExtractionRegion(desiredRegion3)
    slice3.SetInput(rescale_filter.GetOutput())
    slice3.Update()

    self.rep_log.log("[DRR] Extract image filter")

    ######################################
    # Save as NRRD again
    ######################################
    output_path = "output_fixed2.nrrd"
    itk.imwrite(slice3.GetOutput(), output_path)

    self.rep_log.log("[DRR] Save as NRRD (second file)")

    # im1 = itk.array_from_image(resample_filter.GetOutput())
    # print("IM1 shape: ", im1.shape)

    projectionArray = itk.array_from_image(rescale_filter.GetOutput())
    self.rep_log.log("IM2 shape: ", projectionArray.shape)

    # projectionArray = itk.array_from_image(slice3.GetOutput())
    # print("IM3 shape: ", projectionArray.shape)

    ######################################
    # Update Output Volume from array
    ######################################
    self.rep_log.log("[GENERATE-DRR] Updating Output volume Node...")
    slicer.util.updateVolumeFromArray(outputVolumeNode, projectionArray)

    return projectionArray

  def getVolumeArrayFromVolumeNode(self, volumeNode):
    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    return volumeArray

  def calcXRayTransformEquation(self, volumeArray, min_, max_, beta):

    volumeTransformed = np.clip(volumeArray, min_, max_) + min_

    volumeTransformed = beta * (volumeTransformed) / 1000

    volumeTransformed = np.exp(volumeTransformed)

    return volumeTransformed

  def calcProjections(self, volumeArray, axes, beta=0.85, isPreCalc=False):

    self.rep_log.log("Starting projections...")

    max_ = 1500
    min_ = -1024

    ## Pixel calculation
    if not isPreCalc:
      self.rep_log.log("Calculating transform...")
      volumeTransformed = self.calcXRayTransformEquation(volumeArray, min_, max_, beta)
    else:
      volumeTransformed = volumeArray
      # volume_transformed = np.clip(volume_array, min_, max_) + min_
      # volume_transformed = beta * (volume_transformed) / 1000
      # volume_exp = np.exp(volume_transformed)

    ## Projection
    projections = []
    for axis in axes:
      print("Projecting axis {}".format(axis))
      aux = np.sum(volumeTransformed, axis=axis)
      aux = np.expand_dims(aux, -1)
      projections.append(aux)

    projections = np.array(projections)

    self.rep_log.log("Projections Done.")

    return projections

  #----------------------------------------------------
  # Check if Target has been reached with needle
  #----------------------------------------------------
  def isNeedleTipInTargetArea(self, updateNeedleTransform=True):
    isTargetReachedGreenArea = self.targetReachedGreenAreaBreachWarningNode.IsToolTipInsideModel()
    isTargetReachedYellowArea = self.targetReachedYellowAreaBreachWarningNode.IsToolTipInsideModel()

    if isTargetReachedGreenArea:
      isTargetReached = "GreenArea"
    elif isTargetReachedYellowArea:
      isTargetReached = "YellowArea"
    elif isTargetReachedGreenArea and isTargetReachedYellowArea:
      isTargetReached = "GreenArea"
    else:
      isTargetReached = "RedArea"

    if updateNeedleTransform:
      needlePositionTransform = self.getModelPositionTransform(self.needleModelNode)
      matrixArray = self.utils.getMatrixArrayFromTransformNode(needlePositionTransform)
      self.updateDATA("NeedlePositionTransformsAtTargetReached", matrixArray)

    return isTargetReached

  #----------------------------------------------------
  # Others
  #----------------------------------------------------
  def calculateSurgicalTimeFromProjections(self, numProjections, projectionsComputationalTime, repetitionTotalTime):

    singleProjectionRealTime = 5  # seconds

    return singleProjectionRealTime * numProjections - projectionsComputationalTime + repetitionTotalTime

  def copyAndHardenModel(self, originalModelNode):
    ## 1. Clone Model
    outputModel = self.cloneNode(originalModelNode)
    outputModel.GetDisplayNode().SetVisibility(0)

    ## 2. Get Harden transform from model
    parentTransform = originalModelNode.GetParentTransformNode()
    temporalTransform = self.utils.getOrCreateTransform("TemporalTransform")
    self.makeTransformIdentity(temporalTransform)
    temporalTransform.SetAndObserveTransformNodeID(parentTransform.GetID())
    temporalTransform.HardenTransform()

    ## 3. Get Model under Tree tranform and harden
    outputModel.SetAndObserveTransformNodeID(temporalTransform.GetID())
    outputModel.HardenTransform()

    return outputModel, temporalTransform

  def getModelPositionTransform(self, originalModelNode):
    ## 1. Get Harden transform from model
    parentTransform = originalModelNode.GetParentTransformNode()
    temporalTransform = self.utils.getOrCreateTransform("TemporalTransform")
    self.makeTransformIdentity(temporalTransform)
    temporalTransform.SetAndObserveTransformNodeID(parentTransform.GetID())
    temporalTransform.HardenTransform()

    return temporalTransform

  def cloneNode(self, associatedDataNode):
    ## IT HAS A BUG, DO NOT USE UNTIL FIXED
    # shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    # itemIDToClone = shNode.GetItemByDataNode(nodeToClone)
    # clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
    # clonedNode = shNode.GetItemDataNode(clonedItemID)
    #
    # return clonedNode

    ## Create Data Node Clone
    clonedDataNode = slicer.mrmlScene.CreateNodeByClass(associatedDataNode.GetClassName())
    clonedDataNodeName = associatedDataNode.GetName() + "Clone"
    slicer.mrmlScene.AddNode(clonedDataNode)

    ## CLone Display Node
    clonedDisplayNode = slicer.mrmlScene.CreateNodeByClass(associatedDataNode.GetDisplayNode().GetClassName())
    clonedDisplayNode.CopyContent(associatedDataNode.GetDisplayNode())
    # clonedDisplayNode.setName(clonedDataNodeName + "_Display")
    slicer.mrmlScene.AddNode(clonedDisplayNode)

    ## Clone Storage Node
    clonedStorageNode = slicer.mrmlScene.CreateNodeByClass(associatedDataNode.GetStorageNode().GetClassName())
    clonedStorageNode.CopyContent(associatedDataNode.GetStorageNode())
    # clonedStorageNode.setName(associatedDataNode.GetStorageNode().GetFileName() + "_Display")
    slicer.mrmlScene.AddNode(clonedStorageNode)

    ## Copy Data Node
    clonedDataNode.CopyContent(associatedDataNode)
    clonedDataNode.SetName(clonedDataNodeName)
    clonedDataNode.SetAndObserveDisplayNodeID(clonedDisplayNode.GetID())
    clonedDataNode.SetAndObserveStorageNodeID(clonedStorageNode.GetID())

    return clonedDataNode

  def startSimulationRepetition(self, selectedTargetForamen):
    self.DATA_DICT = self.createRepetitionDataDict()

    ## Load target models (green and yellow) and breach warnings
    name = "TargetModelGreenArea_{}".format(selectedTargetForamen)
    path_aux = os.path.join(self.phantomData_path, "TargetModel_GreenArea_{}.stl".format(selectedTargetForamen))
    self.targetModelGreenAreaNode = self.utils.loadModelFromFile(name, path_aux, color=[0, 1, 0], visibility_bool=False, opacity=0.4)

    name = "TargetModelYellowArea_{}".format(selectedTargetForamen)
    path_aux = os.path.join(self.phantomData_path, "TargetModel_YellowArea_{}.stl".format(selectedTargetForamen))
    self.targetModelYellowAreaNode = self.utils.loadModelFromFile(name, path_aux, color=[1, 1, 0], visibility_bool=False, opacity=0.4)

    self.targetReachedGreenAreaBreachWarningNode = self.getOrCreateBreachWarningNode(
      "TargetReachedGreenAreaBreachWarning", self.targetModelGreenAreaNode, self.NeedleTipToNeedle)
    self.targetReachedYellowAreaBreachWarningNode = self.getOrCreateBreachWarningNode(
      "TargetReachedYellowAreaBreachWarning", self.targetModelYellowAreaNode, self.NeedleTipToNeedle)

  def makeNewDir(self, path):
    try:
      os.makedirs(path)
      print("Directory created: %s" % path)
    except FileExistsError:
      print("Directory already exists: %s" % path)

  def setDRRParams(self, DRRParamsMatrix=None, sid=400, drrthreshold=-50, drrsizex=512, drrsizey=512):
    DRRParams = {}

    ## generateDRR my formula
    DRRParams["min"] = -1024
    DRRParams["max"] = 1500
    DRRParams["beta"] = 0.8

    ## generateDRR based on Slicer Module
    translation, rotation = self.getTranslationAndRotationFromMatrixArray(DRRParamsMatrix)
    DRRParams["translation"] = translation
    DRRParams["rot"] = rotation
    DRRParams["center"] = np.zeros(3)
    DRRParams["drrthreshold"] = drrthreshold
    DRRParams["sid"] = sid
    DRRParams["drrsizex"] = drrsizex
    DRRParams["drrsizey"] = drrsizey

    return DRRParams

  def getTranslationAndRotationFromMatrixArray(self, transformMatrix):
    if transformMatrix is None:
      translation = np.zeros(3)
      rotation = np.zeros(3)
    else:
      # matrix = self.utils.getMatrixArrayFromTransformNode(transformMatrix)

      ## Get Translation
      translation = [-transformMatrix[0, 3], -transformMatrix[1, 3], -transformMatrix[2, 3]]

      ## Get Rotation
      r = R.from_matrix(transformMatrix[:3, :3])
      r_values = r.as_euler('zyx', degrees=True)
      rotation = [r_values[2], r_values[1], r_values[0]]

    return translation, rotation

  def getOrCreateBreachWarningNode(self, nodeName, targetModelNode, trandformNode):
    try:
      breachWarningNode = slicer.util.getNode(nodeName)
      breachWarningNode.SetOriginalColor(targetModelNode.GetDisplayNode().GetColor())
      breachWarningNode.SetAndObserveWatchedModelNodeID(targetModelNode.GetID())
      breachWarningNode.SetAndObserveToolTransformNodeId(trandformNode.GetID())
    except:
      breachWarningNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLBreachWarningNode')
      breachWarningNode.UnRegister(None)
      breachWarningNode.SetName(nodeName)
      slicer.mrmlScene.AddNode(breachWarningNode)
      breachWarningNode.SetPlayWarningSound(False)
      breachWarningNode.SetWarningColor(1, 0, 0)
      breachWarningNode.SetOriginalColor(targetModelNode.GetDisplayNode().GetColor())
      breachWarningNode.SetAndObserveWatchedModelNodeID(targetModelNode.GetID())
      breachWarningNode.SetAndObserveToolTransformNodeId(trandformNode.GetID())
      breachWarningNode.SetPlayWarningSound(False)
      breachWarningNode.SetDisplayWarningColor(False)

    return breachWarningNode

  ##----------- SAVING FUNCTIONS ---------- ##
  def saveRepetitionData(self, phantomID, userID, repetitionID, savePath, targetSelected):

    ## 1. Create repetition folder
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    rep_path = os.path.join(savePath, "RecordedResults", "TraditionalMethod", phantomID, "User_{}".format(userID),
                            "Rep_{}_{}_{}".format(repetitionID, targetSelected, date))
    self.makeNewDir(rep_path)

    ## 2. Save Statistical results
    self.saveStatisticalResults(rep_path, phantomID, userID, repetitionID)

    ## 3. Save Projections
    self.saveProjections(rep_path, phantomID, userID, repetitionID)

    ## 4. Save needle position transform
    self.saveNeedlePositionPerProjection(rep_path)
    self.saveNeedlePositionPerTargetReached(rep_path)

    ## 5. Copy Log file to folder rep
    rep_log_path = os.path.join(rep_path, self.rep_log.log_file_name)
    shutil.copyfile(self.rep_log.log_file_path, rep_log_path)

    pass

  def updateDATA(self, key, value):
    if key == "TimePerProjection":
      self.DATA_DICT["TimePerProjection"].append(value)
    elif key == "TimeAtEachProjection":
      self.DATA_DICT["TimeAtEachProjection"].append(value)
    elif key == "ComputationalTimePerProjection":
      self.DATA_DICT["ComputationalTimePerProjection"].append(value)
    elif key == "Projections":
      self.DATA_DICT["Projections"].append(value)
    elif key == "NeedlePositionTransforms":
      self.DATA_DICT["NeedlePositionTransforms"].append(value)
    elif key == "OutputPerTargetReachedButtonClicked":
      self.DATA_DICT["OutputPerTargetReachedButtonClicked"].append(value)
    elif key == "TimeAtEachTargetReachedButtonClicked":
      self.DATA_DICT["TimeAtEachTargetReachedButtonClicked"].append(value)
    elif key == "NeedlePositionTransformsAtTargetReached":
      self.DATA_DICT["NeedlePositionTransformsAtTargetReached"].append(value)
    else:
      self.DATA_DICT[key] = value

  def createRepetitionDataDict(self):
    DATA_DICT = {}

    DATA_DICT["RepetitionTotalTime"] = 0.0  # Total time from start to stop
    DATA_DICT["NumberOfProjections"] = int  # Number of projection images taken during repetition
    DATA_DICT["EstimatedSurgicalTime"] = 0.0  # Estimated surgical time (depending on number of projections)
    DATA_DICT["TimePerProjection"] = []  # Time passed between each projection
    DATA_DICT["TimeAtEachProjection"] = []  # Time when the projection button was clicked
    DATA_DICT["ComputationalTimePerProjection"] = []  # Time the computer took to estimate the projection
    DATA_DICT["NumberOfPunctures"] = 0  # Number of punctures done (there could be more punctures than projections)
    DATA_DICT["NumberOfTimesTargetReachedButtonClicked"] = 0  # Number of times the target reached was clicked
    DATA_DICT["OutputPerTargetReachedButtonClicked"] = []  # The results (Gren, Yellow or Red) when target reached button was clicked
    DATA_DICT["TimeAtEachTargetReachedButtonClicked"] = []   # Time at each target reached button was clicked

    DATA_DICT["TargetSelected"] = "None"

    DATA_DICT["Projections"] = []
    DATA_DICT["NeedlePositionTransforms"] = []
    DATA_DICT["NeedlePositionTransformsAtTargetReached"] = []

    return DATA_DICT

  def saveStatisticalResults(self, folder_path, phantomID, userID, repetitionID):
    DATA = {}
    DATA["phantomID"] = [phantomID]
    DATA["userID"] = [userID]
    DATA["repetitionID"] = [repetitionID]

    keys = ["TargetSelected", "RepetitionTotalTime", "NumberOfProjections", "NumberOfPunctures",  "EstimatedSurgicalTime",
            "TimePerProjection", "TimeAtEachProjection", "ComputationalTimePerProjection",
            "NumberOfTimesTargetReachedButtonClicked", "OutputPerTargetReachedButtonClicked", "TimeAtEachTargetReachedButtonClicked"]
    for key in keys:
      DATA[key] = [self.DATA_DICT[key]]

    DATA_pd = pd.DataFrame.from_dict(DATA)

    file_path = os.path.join(folder_path, "StatisticalResults.csv")
    self.rep_log.log("[SAVE-ST] Saving statistics to {}".format(file_path))
    pd.DataFrame.to_csv(DATA_pd, file_path, index=False)

  def saveProjections(self, folder_path, phantomID, userID, repetitionID):
    projectionFolderPath = os.path.join(folder_path, "Projections")
    self.makeNewDir(projectionFolderPath)

    ## Save numpy
    file_path = os.path.join(projectionFolderPath, "ProjectionsData_{}_{}_{}".format(phantomID, userID, repetitionID))
    npArray = np.array(self.DATA_DICT["Projections"])
    self.rep_log.log("[SAVE-PRJ] Saving projections to {}".format(file_path))
    np.save(file_path, npArray)

    ## Save each projectino as PNG
    for i in range(npArray.shape[0]):
      nameImageFile = "Projection_{}.png".format(i+1)
      file_path = os.path.join(projectionFolderPath, nameImageFile)
      self.saveProejctionAsImage(npArray[i,0,:,:], file_path)

  def saveProejctionAsImage(self, x, fig_save_path, vmin=None, vmax=None):
    vmin = x.min() if vmin is None else vmin
    vmax = x.max() if vmax is None else vmax

    fig = plt.figure(figsize=(7, 7))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x, cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')

    fig.savefig(fig_save_path, transparent=True)

    plt.close(fig)

  def saveNeedlePositionPerProjection(self, folder_path):
    needlePositionTransformFolderPath = os.path.join(folder_path, "NeedlePositionTransformsPerProjection")
    self.makeNewDir(needlePositionTransformFolderPath)

    key = "NeedlePositionTransforms"
    self.DATA_DICT[key] = np.array(self.DATA_DICT[key])

    ## Create temporal transform file
    transformNode = self.utils.getOrCreateTransform("TemporalTransform")
    transformNode.SetAndObserveTransformNodeID(None)

    ## Save each transform as individual file
    for i in range(self.DATA_DICT[key].shape[0]):
      # 1. Make Transform Identity
      self.makeTransformIdentity(transformNode) ## TODO: puede que sea inutil

      ## Copy transform
      temporalMatrix = self.utils.vtkMatrixFromArray(self.DATA_DICT[key][i])
      transformNode.SetMatrixTransformToParent(temporalMatrix)

      ## Save
      file_name = 'NeedlePositionInProjection_{}_Transform.h5'.format(i+1)
      file_path = os.path.join(needlePositionTransformFolderPath, file_name)
      self.utils.saveDataWithNode(transformNode, file_path)

    self.rep_log.log("[SAVE-NEEDLEPOS] Needle positions transforms SAVED.")
    pass

  def saveNeedlePositionPerTargetReached(self, folder_path):
    needlePositionTransformFolderPath = os.path.join(folder_path, "NeedlePositionTransformsPerTargetReached")
    self.makeNewDir(needlePositionTransformFolderPath)

    key = "NeedlePositionTransformsAtTargetReached"
    self.DATA_DICT[key] = np.array(self.DATA_DICT[key])

    ## Create temporal transform file
    transformNode = self.utils.getOrCreateTransform("TemporalTransform")
    transformNode.SetAndObserveTransformNodeID(None)

    ## Save each transform as individual file
    for i in range(self.DATA_DICT[key].shape[0]):
      # 1. Make Transform Identity
      self.makeTransformIdentity(transformNode) ## TODO: puede que sea inutil

      ## Copy transform
      temporalMatrix = self.utils.vtkMatrixFromArray(self.DATA_DICT[key][i])
      transformNode.SetMatrixTransformToParent(temporalMatrix)

      ## Save
      file_name = 'NeedlePositionInTargetReached_{}_Transform.h5'.format(i+1)
      file_path = os.path.join(needlePositionTransformFolderPath, file_name)
      self.utils.saveDataWithNode(transformNode, file_path)

    self.rep_log.log("[SAVE-NEEDLEPOSTR] Needle positions transforms SAVED.")
    pass

  def makeTransformIdentity(self, transformNode):
    identityTransform = vtk.vtkMatrix4x4()
    transformNode.SetMatrixTransformToParent(identityTransform)

  def recordSoftwareActivity(self, actionName):

    # Store actionName
    self.recordedActivity_action.append(actionName)

    # Store timestamp
    self.recordedActivity_timeStamp.append(time.strftime("%Y-%m-%d_%H-%M-%S"))

  def saveSoftwareActivity(self):

    dateAndTime = time.strftime("_%Y-%m-%d_%H-%M-%S")
    csvFilePath = self.CRS_record_path + 'RecordedActivity_' + dateAndTime + '.csv'

    with open(csvFilePath, 'wb') as csvfile:
      writer = csv.writer(csvfile, delimiter=",")
      writer.writerow(['timestamp', 'action'])

      timestamp_array = np.array(self.recordedActivity_timeStamp)
      action_array = np.array(self.recordedActivity_action)

      for i in range(timestamp_array.shape[0]):
        vector = np.array([timestamp_array[i], action_array[i]])
        writer.writerow(vector)

  ## WHATCHDOG
  def addWatchdog(self, transformNode, watchedNodeID, warningMessage, playSound):
    """
    Function to add watchdog node to a transformation node. A warning message will be shown on screen when the tool is out of view.
    """
    self.wd.AddWatchedNode(transformNode)
    self.wd.SetWatchedNodeWarningMessage(watchedNodeID, warningMessage)
    self.wd.SetWatchedNodeUpdateTimeToleranceSec(watchedNodeID, 0.2)
    self.wd.SetWatchedNodePlaySound(watchedNodeID, playSound)

  def removeAllWatchedNodes(self):
    self.wd = slicer.util.getNode('WatchdogNode')
    self.wd.RemoveAllWatchedNodes()

class Utils():

  def __init__(self):
    pass

  def saveData(self, node_name, file_path, file_name):
    # Save node to path
    node = slicer.util.getNode(node_name)
    path = os.path.join(file_path, file_name)
    print("[SAVE-DATA] Node {} has been saved in path: {}".format(node_name, path))
    return slicer.util.saveNode(node, path)

  def saveDataWithNode(self, node, file_path):
    # Save node to path
    node_name = node.GetName()
    print("[SAVE-DATA] Node {} has been saved in path: {}".format(node_name, file_path))
    return slicer.util.saveNode(node, file_path)

  def getOrCreateTransform(self, transformName):
    """
    Gets existing tranform or create new transform containing the identity matrix.
    """
    try:
      transformNode = slicer.util.getNode(transformName)
    except:
      print('ERROR: Transformation node {} was not found. Creating node as identity...'.format(transformName))
      transformNode = slicer.vtkMRMLLinearTransformNode()
      transformNode.SetName(transformName)
      slicer.mrmlScene.AddNode(transformNode)

    return transformNode

  def getOrCreateFiducials(self, fiducialsName, color=None, visibility_bool=False):
    """
    Gets existing point set or create a new point set.
    """
    if color is None:
      color = [0, 0, 0]

    try:
      fiducialNode = slicer.util.getNode(fiducialsName)
    except:
      print('ERROR: Transformation node {} was not found. Creating node as identity...'.format(fiducialsName))
      fiducialNode = slicer.vtkMRMLMarkupsFiducialNode()
      fiducialNode.SetName(fiducialsName)
      slicer.mrmlScene.AddNode(fiducialNode)
      # fiducialNode.GetDisplayNode().SetSelectedColor(color)
      fiducialNode.GetDisplayNode().SetVisibility(visibility_bool)
      fiducialNode.LockedOn()
    return fiducialNode

  def createLabelMapVolumeNode(self, volumeName):
    newVolumeNode = slicer.vtkMRMLLabelMapVolumeNode()
    newVolumeNode.SetName(volumeName)
    slicer.mrmlScene.AddNode(newVolumeNode)
    return newVolumeNode

  def getOrCreateVolume(self, volumeName):
    try:
      volumeNode = slicer.util.getNode(volumeName)
    except:
      print('ERROR: Volume node {} was not found. Creating empty node...'.format(volumeName))
      volumeNode = slicer.vtkMRMLScalarVolumeNode()
      volumeNode.SetName(volumeName)
      slicer.mrmlScene.AddNode(volumeNode)
    return volumeNode

  def loadTransformFromFile(self, transformName, transformFilePath):
    try:
      transformNode = slicer.util.getNode(transformName)
    except:
      try:
        transformNode = slicer.util.loadTransform(transformFilePath)
        print('Transform loaded: {}'.format(transformName))
      except:
        print('ERROR: Transformation node {} was not found. Creating node as identity...'.format(transformName))
        transformNode = slicer.vtkMRMLLinearTransformNode()
        transformNode.SetName(transformName)
        slicer.mrmlScene.AddNode(transformNode)
    return transformNode

  def loadModelFromFile(self, modelName, modelFilePath, color=None, visibility_bool=True, opacity=1.0):
    if color is None:
      color = [0, 0, 0]

    try:
      modelNode = slicer.util.getNode(modelName)
    except:
      try:
        modelNode = slicer.util.loadModel(modelFilePath)

        modelNode.GetModelDisplayNode().SetColor(color)
        modelNode.GetModelDisplayNode().SetVisibility(visibility_bool)
        modelNode.GetModelDisplayNode().SetOpacity(opacity)
        print('Model loaded: {}'.format(modelName))
      except:
        print('ERROR: {} model not found in path: {}'.format(modelName, modelFilePath))
        modelNode = None
    return modelNode

  def loadFiducialsFromFile(self, fiducialsName, fiducialsFilePath, color=None, visibility_bool=False):
    """
    Gets existing point set or create a new point set.
    """
    if color is None:
      color = [0, 0, 0]

    try:
      fiducialNode = slicer.util.getNode(fiducialsName)
    except:
      [success, fiducialNode] = slicer.util.loadMarkupsFiducialList(fiducialsFilePath)
      if success:
        fiducialNode.GetDisplayNode().SetSelectedColor(color)
        fiducialNode.GetDisplayNode().SetVisibility(visibility_bool)
        fiducialNode.LockedOn()
        print('Fiducial List loaded: {}'.format(fiducialsName))
      else:
        print('ERROR: {} fiducials not found in path: {}'.format(fiducialsName, fiducialsFilePath))
        fiducialNode = None
    return fiducialNode

  def loadVolumeFromFile(self, volumeName, volumeFilePath):

    try:
      volumeNode = slicer.util.getNode(volumeName)
    except:
      try:
        volumeNode = slicer.util.loadVolume(volumeFilePath)
      except:
        print('ERROR: {} volume not found in path: {}'.format(volumeName, volumeFilePath))
        volumeNode = None
    return volumeNode

  def vtkMatrixFromArray(self, transformMatrixArray):

    vtkTransform = vtk.vtkMatrix4x4()

    for i in range(4):
      for j in range(4):
        vtkTransform.SetElement(i, j, transformMatrixArray[i, j])

    return vtkTransform

  def ArrayFromVTKMatrix(self, vtkMatrix):

    transformMatrixArray = np.identity(4)

    for i in range(4):
      for j in range(4):
        transformMatrixArray[i,j] = vtkMatrix.GetElement(i, j)

    return transformMatrixArray

  def getMatrixArrayFromTransformNode(self, transformNode):
    vtkTransform = vtk.vtkMatrix4x4()
    transformNode.GetMatrixTransformToParent(vtkTransform)
    transformMatrixArray = self.getMatrixArrayFromVTKMatrix(vtkTransform)

    return transformMatrixArray

  def getMatrixArrayFromVTKMatrix(self, vtkTransform):

    transformMatrixArray = np.identity(4)

    for i in range(4):
      for j in range(4):
        transformMatrixArray[i, j] = vtkTransform.GetElement(i, j)

    return transformMatrixArray

  def setTranslation(self, transform, tx, ty, tz):

    vTransform = vtk.vtkTransform()
    vTransform.Translate(tx, ty, tz)
    transform.SetMatrixTransformToParent(vTransform.GetMatrix())

  def setRotation(self, transform, rx, ry, rz):

    rotMatrix = vtk.vtkTransform()
    rotMatrix.RotateZ(rz)
    rotMatrix.RotateY(ry)
    rotMatrix.RotateX(rx)
    transform.SetMatrixTransformToParent(rotMatrix.GetMatrix())

  def setTranslationAndRotation(self, transform, tx, ty, tz, rx, ry, rz):

    vTransform = vtk.vtkTransform()
    vTransform.RotateZ(rz)
    vTransform.RotateY(ry)
    vTransform.RotateX(rx)

    vMatrix = vTransform.GetMatrix()
    vMatrix.SetElement(0, 3, tx)
    vMatrix.SetElement(1, 3, ty)
    vMatrix.SetElement(2, 3, tz)
    transform.SetMatrixTransformToParent(vMatrix)

  def setTranslationAndRotationToVTK(self, tx, ty, tz, rx, ry, rz):

    vTransform = vtk.vtkTransform()
    vTransform.RotateZ(rz)
    vTransform.RotateY(ry)
    vTransform.RotateX(rx)

    vMatrix = vTransform.GetMatrix()
    vMatrix.SetElement(0, 3, tx)
    vMatrix.SetElement(1, 3, ty)
    vMatrix.SetElement(2, 3, tz)

    return vTransform

class MyLog:
  def __init__(self, log_file_path="my.log", log_name="SlicerModuleLog"):
    self.log_file_path = log_file_path
    self.log_name = log_name
    self.logger = None
    self.log_file_name = self.log_file_path.split("\\")[-1]

  def init_log(self):
    # logging.basicConfig(filename=self.log_file_path, filemode='w',
    #                     format='%(asctime)s: %(message)s', datefmt='%m-%d-%Y_%H:%M:%S',
    #                     level=logging.DEBUG)

    # create logger
    self.logger = logging.getLogger(self.log_name)
    self.logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.FileHandler(self.log_file_path, "w+")
    # ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s_%(name)s: %(message)s', datefmt='%m-%d-%Y_%H:%M:%S')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    self.logger.addHandler(ch)

    return True

  def log(self, text, log_val=0):
    # if log_val:
    #   print(text)

    self.logger.info(text)