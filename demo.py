import os
import time
from pathlib import Path
from typing import List
from shutil import copy2
from dflow import config


from dflow import (
    InputParameter,
    InputArtifact,
    Inputs,
    OutputArtifact,
    Outputs,
    Step,
    Steps,
    Workflow,
    argo_range,
    download_artifact,
    upload_artifact
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
    upload_packages,
)


class MakeLammpsIn(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "templates_dir": Artifact(Path),
                "np_lammps": int,
                "thermostat_t": float,
                "ion_energy": float,
                "n2i_ratio": float,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({"lammpsin": Artifact(Path)})

    @staticmethod
    def processor_mapping(np):
        if np == 128:
            return "8 8 2"
        elif np == 64:
            return "8 8 1"
        elif np == 32:
            return "4 4 2"
        elif np == 16:
            return "4 4 1"
        elif np == 1:
            return "1 1 1"
        else:
            raise RuntimeError(f"{np} processor used but idk how to divide the box")

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        def filetostring(fname):
            with open(fname) as fhandle:
                string = "".join(fhandle.readlines())
            return string

        fname = op_in["templates_dir"] / "main.lmp"

        shift_before = op_in["templates_dir"] / "loop_shift_before_run.lmp"
        shift_after = op_in["templates_dir"] / "loop_shift_after_run.lmp"

        fix_deposit_neutral = "fix new_neu ion_and_neu deposit ${N_neu} 2 ${Dt_n} 1 region top_depo near 2 vz ${vzn} ${vzn} vx -${vxy} ${vxy} vy -${vxy} ${vxy}"

        # do you have fast ions?
        if op_in['ion_energy'] == 0.0:
            fix_deposit_ion = "fix new_ion ion_and_neu deposit ${N_ion} 2 ${Dt_i} 1 region top_depo near 2 vz ${vz} ${vz}"
            run = op_in["templates_dir"] / "loop_run_0eV.lmp"
        else:
            fix_deposit_ion = "fix new_ion ion_and_neu deposit ${N_ion} 2 ${Dt_i} 1 region top_depo near 2 vz ${vzn} ${vzn} vx -${vxy} ${vxy} vy -${vxy} ${vxy}"
            run = op_in["templates_dir"] / "loop_run_not0eV.lmp"
            
        # main script specifies all the settings
        main_template = filetostring(fname)
        main_formatted = main_template.format(
            proc_map=MakeLammpsIn.processor_mapping(op_in["np_lammps"]),
            ion_energy=op_in["ion_energy"],
            thermostat_t=op_in["thermostat_t"],
            n2i_ratio=op_in["n2i_ratio"],
            fix_deposit_ion=fix_deposit_ion,
            fix_deposit_neutral=fix_deposit_neutral,
        )

        # copy files over
        to_copy = [shift_before, shift_after, run]
        fnames = [
            "loop_shift_before_run.lmp",
            "loop_shift_after_run.lmp",
            "loop_run.lmp",
        ]

        for tfname, ofname in zip(to_copy, fnames):
            if not Path(ofname).exists():
                copy2(tfname, Path(ofname))

        # write main script
        mainfile = Path("main.lmp")
        if not mainfile.exists():
            mainfile.write_text(main_formatted)

        op_out = OPIO(
            {
                "lammpsin": mainfile,
            }
        )

        return op_out


class Dummy(OP):
    """Meant to be an analysis step: analysis the dump
    trajectory file for film thickness and oxygen content
    """
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
        })
    
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "thickness": Artifact(Path),
            "o_content": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        thickness = Path("thickness.data")
        o_content = Path("o_content.data")

        ofiles = [thickness, o_content]
        
        for ofile in ofiles:
            ofile.write_text('dummy')

        op_out = OPIO(
            {
                "thickness": thickness,
                "o_content": o_content
             }
        )

        return op_out


def make_lammps_wf(np, thermostat_t, ion_energy, n2i_ratio, templates_dir):
    wf = Workflow(name="lammps")

    artifact0 = upload_artifact(templates_dir)

    make_lammpsin = Step(
        name="make-lammpsin",
        template=PythonOPTemplate(MakeLammpsIn,
                                  image="python:3.8",
                                  output_artifact_archive={"lammpsin": None}),
        parameters={
            "np_lammps": np,
            "thermostat_t": thermostat_t,
            "ion_energy": ion_energy,
            "n2i_ratio": n2i_ratio,
        },
        artifacts={
            "templates_dir": artifact0,
        })

    wf.add(make_lammpsin)
    
    dummy_output = Step(name="dummy_output",
                        template=PythonOPTemplate(Dummy,
                                                  image="python3.8",
                                                  output_artifact_archive={"thickness": None,
                                                                           "o_content": None}))
    
    wf.add(dummy_output)
    wf.submit()



    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(4)
    
    assert wf.query_status() == "Succeeded"
    step = wf.query_step(name="dummy_output")[0]
    assert step.phase == "Succeeded"
    download_artifact(step.outputs.artifacts["thickness"])
    download_artifact(step.outputs.artifacts["o_content"])


if __name__ == "__main__":
    config["mode"] = "debug"
    make_lammps_wf(np=128,
                   thermostat_t=173.0,
                   ion_energy=10.0,
                   n2i_ratio=10.0,
                   templates_dir='lmp_templates')
