"""
Filename:MAgent.py
Created Date: Tuesday, Oct 27th 2024, 6:52:25 pm
Author: XiaoQi
"""

'''
icustays_units = ['Medical Intensive Care Unit (MICU)',
 'Surgical Intensive Care Unit (SICU)',
 'Surgical Intensive Care Unit (MICU/SICU)',
 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)',
 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown',
 'Neuro Surgical Intensive Care Unit (Neuro SICU)']
'''
import asyncio
from datetime import datetime
from metagpt.environment import Environment
from metagpt.roles import Role
from metagpt.team import Team
from metagpt.actions import Action, UserRequirement
from pydantic import ConfigDict, Field
from gymnasium import spaces
from metagpt.logs import logger
from typing import Optional
import platform
from typing import Any
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import fire
import json
import re
from metagpt.schema import Message
import git
import torch
import transformers
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class SurgeryPlan(Action):

    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are surgerydoctor, You need to make a detailed surgery plan according given one patient's medical record result and Several similar medical records.
    ## MEDICAL RECORD
    Paper record:
    {context}
    ## YOUR TURN
    Now it's your turn,you will need to write a surgical plan that is as detailed and comprehensive as possible based on the MRI result and Reference surgery plan {context} .
    You need to find the deatures of report and give the surgery method and surgery name based on reference surgery plan.

    return ```Surgery Name: ```
    return ```Tumor Condition: ```
    return ```Surgery Steps: ```

    
    """



    name: str = "Plan"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)
        # logger.info(prompt)

        rsp = await self._aask(prompt)
        list = []
        sub_path = "sub_sub_memories.json"
        # with open(sub_path, "r") as sub_json_file:
        #     existing_data = json.load(sub_json_file)
        # list.append(msg.to_dict())
        list.append(rsp)

        with open(sub_path, "w") as f:
            json.dump(list, f, indent=2)  

        return rsp

class Transfer(Action):
    """Action: Speak out aloud in a debate (quarrel)"""

    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are {name}, you are collaborating with  {collaborator_name} to finish transfering patient from ward to operation room.
    ## SURGERY HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn, if you you should closely respond to your {collaborator_name}'s latest requirement, You can proceed to the next conversation only after you receive new information.
    Both of you need to complete each step of the tansfer in a conversational manner with each other step by step.
    if you are Wardnurse, your task is to transfer patient to operation room and introduce patient's condition. You can proceed to the next conversation only after you receive new information.
    if you are Roomnurse,your task is mainly focus on receive patient and check wardnurseConfirm some key information about the patient. You can proceed to the next conversation only after you receive new information.
    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.


    when finished the transfer patient into the operation room,stop
    """
    name: str = "Transfer"

    async def run(self, context: str, name: str, collaborator_name: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, collaborator_name=collaborator_name)
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp
    


class Mazui(Action):


    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are {name}, you are collaborating with  {collaborator_name} to finish Anaesthetization.
    ## SURGERY HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn, if you you should closely respond to your {collaborator_name}'s latest requirement,  you can only just say once You can proceed to the next conversation only after you receive new information.
    You need to complete each step of the check in a conversational manner with each other step by step.
    if you are Anesthesiologist, your task is to check patient anesthesia-related issues (drinking water, eating, etc., such as fasting for 8 hours) with Roomnurse.
    return ```Operation:[] ```
    return ```Anesthesiologist:"" ```
    if you are Roomnurse,your task is mainly focus on Assist in confirming the patient's condition and help proceed anesthesia.
    return ```Operation:[] ```
    return ```Roomnurse:"" ```
    """
    name: str = "Mazui"

    async def run(self, context: str, name: str, collaborator_name: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, collaborator_name=collaborator_name)
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp

class ChangeTools(Action):

    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are {name}, you are collaborating with  {collaborator_name} to finish the whole surgery.
    ## SURGERY HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn, if you you should closely respond to your {collaborator_name}'s latest requirement, 
    You need to complete each step of the provide equipment in a conversational manner with each other step by step.
    if you are travellingnurse, your task is to provide the surgical tools for every step and check if  need additional equipment.
    return ```Tools check:[] ```
    return ```Travellingnurse:"" ```
    if you are surgeryassistant,your task is mainly focus on assist in completing the operation and ask for surgical instruments and sterilize them.
    return ```Equipment request:[] ```
    return ```Surgeryassistant:"" ```
    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.

    when finished the surgery,stop
    """
    name: str = "ChangeTools"

    async def run(self, context: str, name: str, collaborator_name: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, collaborator_name=collaborator_name)
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp


class Cooperation(Action):

    PROMPT_TEMPLATE_SOLVER: str = """
    ## BACKGROUND
    Suppose you are {name} in a Team of three, you are collaborating on an operation with  {collaboratorA_name},and {collaboratorB_name} based on surgical plan.
    ## SURGERY PLAN
    Step by step(If you encounter a multiple-choice question in the step, choose one at random and use your ability to reason about its consequences.):
    {context}

    ## YOUR TURN
    Now it's your turn, if you you should closely follow the SURGERY PLAN to give  {collaboratorA_name} instruction step by step.

    if you are Solver{name},your task is mainly focus on analyze the situation of the previous surgical step, provide a plan for the next surgical step by supervising  {collaboratorA_name} and {collaboratorB_name}' action. If a doctor's error is observed, remedial methods and follow-up procedures will be provided.
    In surgery ,Check whether the tumor was incompletely removed or cerebrospinal fluid leaked due to incorrect operations during the surgeon's steps.
    return ```Analyse last step:[] ```
    return ```Next step plan:[] ```
    
    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.

    """

    PROMPT_TEMPLATE_DOCTOR: str = """
    ## BACKGROUND
    Suppose you are {name} in a Team of three, you are collaborating on an operation with  {collaboratorA_name},and {collaboratorB_name} based on surgical plan.
    ## SURGERY HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn,you are SurgeryDoctor{name} you should closely respond to your {collaboratorA_name} and {collaboratorB_name}'s latest requirement, 
    You need to complete each step of the operation in a conversational manner with other two persons step by step.
    if you are SurgeryDoctor,your task is mainly focus on finish the key surgical steps detailed by solver{collaboratorA_name}'s next step plan and you may make a mistake in operating.
    return ```Surgeon operating:[] ```
    return ```SurgeryDoctor:"" ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
"""
    PROMPT_TEMPLATE_Nurse: str = """
    ## BACKGROUND
    Suppose you are {name} in a Team of three, you are collaborating on an operation with  {collaboratorA_name},and {collaboratorB_name} based on surgical plan.
    ## SURGERY HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn,you are SurgeryDoctor you should closely respond to your {collaboratorA_name} and {collaboratorB_name}'s latest requirement, 
    You need to complete each step of the operation in a conversational manner with other two persons step by step.
    if you are Nurse, your task is to provide the surgical tools or do some easy surgical steps to help {collaboratorA_name}.
    return ```Surgeon operating:[] ```
    return ```Nurse:"" ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str,collaboratorB_name: str):
        if name == "Solver":
            prompt = self.PROMPT_TEMPLATE_SOLVER.format(context=context, name=name, collaboratorA_name=collaboratorA_name,collaboratorB_name=collaboratorB_name)
        elif name == "SurgeryDoctor":
            prompt = self.PROMPT_TEMPLATE_DOCTOR.format(context=context, name=name, collaboratorA_name=collaboratorA_name,collaboratorB_name=collaboratorB_name)
        elif name == "Nurse":
            prompt = self.PROMPT_TEMPLATE_Nurse.format(context=context, name=name, collaboratorA_name=collaboratorA_name,collaboratorB_name=collaboratorB_name)

        rsp = await self._aask(prompt)

        return rsp


class Review(Action):

    PROMPT_TEMPLATE: str = """

    {context}

    """

    name: str = "Review"

    async def run(self,context: str,report:str):
        prompt = self.PROMPT_TEMPLATE.format(context=context,report=report)
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp