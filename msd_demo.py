from typing import Optional
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    Simulator,
    VectorLogSink,
    RigidTransform,
    StartMeshcat,
    Rgba,
)
from pydrake.geometry import Sphere, Cylinder, Box
from pydrake.math import RotationMatrix
from pydrake.systems.primitives import ConstantVectorSource

# =============================================================================
# PARAMETERS
# =============================================================================
class MSDParams:
    """Physical parameters for simple mass spring damper system"""
    def __init__(self):
        self.m: float = 3.6     # mass (kg)
        self.k: float = 400.0   # spring constant (N/m)
        self.c: float = 10.0    # damping coefficient (N·s/m)

class VisualizationParams:
    """
    Purely visual parameters for the mass–spring–damper Meshcat scene.
    No physics here.
    """
    def __init__(self):
        # --- Layout ---
        self.rail_length = 2.5          # Length of rail (m)
        self.spring_offset_y = 0.10     # Y-offset of spring (m)
        self.damper_offset_y = -0.10    # Y-offset of damper (m)

        # --- Sizes ---
        self.wall_thickness = 0.1
        self.wall_height = 0.4
        self.wall_depth = 0.4

        self.rail_height = 0.05
        self.rail_depth = 0.05

        self.mass_size = 0.18           # Cube side length

        self.spring_radius = 0.015
        self.damper_radius = 0.020

        # --- Colors (RGBA) ---
        self.wall_color   = (0.3, 0.3, 0.3, 1.0)
        self.rail_color   = (0.6, 0.6, 0.6, 1.0)
        self.mass_color   = (0.2, 0.4, 0.9, 1.0)
        self.spring_color = (0.9, 0.7, 0.2, 1.0)
        self.damper_color = (0.7, 0.2, 0.2, 1.0)

        self.update_rate = 30.0 # Hz


# =============================================================================
# PLANT: MSD FROM SCRATCH
# =============================================================================

class SimpleMSD(LeafSystem):
    """
    Mass Spring Damper dynamics:
        state  = [x, xd]
        Dynamics: 
            xdd = -(c/m)*xd -(k/m)*x
    """
    def __init__(self, params):
        super().__init__()
        self.set_name("SimpleMSD")

        self.params = params

        self.DeclareContinuousState(2)

        self.DeclareVectorOutputPort(
            "state", BasicVector(2), self.CopyStateOut,
            prerequisites_of_calc={self.xc_ticket()}
        )

    def CopyStateOut(self, context, output):
        X = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(X)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        x, xd = context.get_continous_state

        m = self.params.m
        k = self.params.k
        c = self.params.c

        xdd = -(c/m)*xd -(k/m)*x

        derivatives.get_mutable_vector().SetFromVector([xd, xdd])
    
# =============================================================================
# VISUALIZER: MESHCAT SCENE BUILT FROM SCRATCH
# =============================================================================
class MSDVisualiser(LeafSystem):
    """
    Takes mass state [x, xdot] and publishes Meshcat transforms.
    """
    def __init__(self, meshcat, plant, viz_params):
        super().__init__()
        self.set_name("MSDVisualiser")

        self.meshcat = meshcat
        self.plant = plant
        self.params = plant.params
        self.viz = viz_params

        # Input: state [x, xd]
        self.DeclareVectorInputPort("state, 2")

        # Periodic publish to update transforms
        self.DeclarePeriodicPublishEvent(
            period_sec= 1.0 / self.viz.update_rate,
            offset_sec=0.0,
            publish= self.UpdateVisualisation
        )

        # Build Meshcat objects (once)
        self.meshcat.SetObject(
            "msd/wall",
            Box(
                self.viz.wall_thickness,
                self.viz.wall_depth,
                self.viz.wall_height
            ),
            Rgba(*self.viz.wall_color)
        )

        # Wall center slightly behind x=0 so face is at x=0
        self.meshcat.SetTransform(
            "msd/wall",
            RigidTransform([
                -self.viz.wall_thickness / 2.0,
                0.0,
                self.viz.wall_height / 2.0
            ])
)
