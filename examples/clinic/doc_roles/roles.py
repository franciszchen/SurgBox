"""
Filename:MAgent.py
Created Date: Tuesday, Oct 27th 2024, 6:52:25 pm
Author: XiaoQi
"""
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
from doc_actions.actions import *


sub_memory_path = "./clinic/data/sub_memories.json"
dynamic_memory_path = "./data/dynamic_memories.json"
static_memory_path = "./data/static_memories.json"

class PlanMaker(Role):
    name: str = "PlanMaker"
    profile: str = "Plan"


    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([SurgeryPlan])
        self._watch([UserRequirement])

async def Plan(idea: str, investment: float = 3.0, n_round: int = 1):

    SurgeryPlan = PlanMaker(name="SurgeryPlan", profile="plan")

    team = Team()
    team.hire([SurgeryPlan])
    team.invest(investment)   
    team.run_project(idea, send_to="SurgeryPlan")  
    await team.run(n_round=n_round)

class Nurse(Role):
    name: str = ""
    profile: str = ""
    collaborator_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Transfer])
        self._watch([Transfer])

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news =   [msg for msg in self.rc.news if msg.send_to == {self.name}]

        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo 

        memories = self.get_memories()
        # print(f"memories type~~~:{memories}")
        # context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
        # print(context)

        rsp = await todo.run(context=memories, name=self.name, collaborator_name=self.collaborator_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaborator_name,
        )
        self.rc.memory.add(msg)

        return msg

class MazuiRole(Role):
    name: str = ""
    profile: str = ""
    collaborator_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Mazui])
        self._watch([Mazui])

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news =   [msg for msg in self.rc.news if msg.send_to == {self.name}]

        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo 

        memories = self.get_memories()
        # print(f"memories type~~~:{memories}")
        # context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
        # print(context)

        rsp = await todo.run(context=memories, name=self.name, collaborator_name=self.collaborator_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaborator_name,
        )
        self.rc.memory.add(msg)

        return msg

class Tools(Role):
    name: str = ""
    profile: str = ""
    collaborator_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([ChangeTools])
        self._watch([ChangeTools])

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news =   [msg for msg in self.rc.news if msg.send_to == {self.name}]

        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  

        memories = self.get_memories()
        # print(f"memories type~~~:{memories}")
        # context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
        # print(context)

        rsp = await todo.run(context=memories, name=self.name, collaborator_name=self.collaborator_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaborator_name,
        )
        self.rc.memory.add(msg)

        return msg
    
class Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    collaboratorB_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Cooperation])
        self._watch([Cooperation])

    async def _observe(self) -> int:
        await super()._observe()

        if self.name == "SurgeryDoctor":
            self.rc.news =   [msg for msg in self.rc.news if msg.send_to == "SurgeryDoctor" or "Solver"]
        elif self.name == "Nurse":
            self.rc.news =   [msg for msg in self.rc.news if msg.send_to == "Nurse" or "Solver"]
        elif self.name == "Solver":
            self.rc.news =   [msg for msg in self.rc.news if msg.send_to == "Solver" or "SurgeryDoctor"]

        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # An instance of SpeakAloud

        memories = self.get_memories()

        rsp = await todo.run(context=memories, name=self.name, collaboratorA_name=self.collaboratorA_name,collaboratorB_name=self.collaboratorB_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaboratorA_name,
        )

        file_path = "sub_memories.json"
        with open(file_path, "r") as json_file:

            existing_data = json.load(json_file)
        # list.append(msg.to_dict())
        existing_data.append(msg.to_dict())

        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=2)  
               

        self.rc.memory.add(msg)


        return msg

class ReviewDoctor(Role):
    name: str = ""
    profile: str = ""
    collaborator_name: str = ""
    report:str = ""



    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Review])
        self._watch([UserRequirement])


    async def _act(self) -> Message:

        todo = self.rc.todo  
        # 读取JSON文件
        with open('sub_memories.json', 'r') as file:
            data = json.load(file)
        result = [f"{item['role']}:{item['content']}" for item in data]

        context = ' '.join(result)





        rsp = await todo.run(context=context,report=self.report)
        print(self.report)
        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaborator_name,
        )
        self.rc.memory.add(msg)
        with open('sub_sub_memories.json', 'r') as file:
            data = json.load(file)    
        for i in data:
            text = i
            start_index = text.find("Surgery Name:") + len("Surgery Name:")

            end_index = text.find("\n", start_index)
            surgery_name = text[start_index:end_index].strip()
        with open('static_memories.json', 'r') as f:
            content = json.load(f)   
        start_index = rsp.find("Grade: ") + len("Grade: ")
        start_index1 = rsp.find("Shortcomings: ") + len("Shortcomings: ")
        end_index = start_index + 1
        end_index1 = text.find("\n", start_index1)     
        grade = rsp[start_index:end_index]
        shortcomings = rsp[start_index1:end_index1]
        new_item = {
        # "unique_id": int(datetime.now().timestamp()),
        # "image_id": item['image_id'],
        "error": surgery_name,
        # "surgery_plan":text,
        # "surgery_process": result,
        "surgery_review":rsp,
        # "grade":grade,
        # "shortcomings":shortcomings
    }
        # with open(save_path, "w") as f:
        # json.dump(res, f, indent=2)
        content.append(new_item)

        with open('static_memories.json', "w") as f:
            json.dump(content, f, indent=2)  
               

        return msg

