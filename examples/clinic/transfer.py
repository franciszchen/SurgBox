
import asyncio

from metagpt.actions import Action
from metagpt.environment import Environment
from metagpt.roles import Role
from metagpt.team import Team
from metagpt.actions import Action, UserRequirement
from pydantic import ConfigDict, Field
from gymnasium import spaces
from metagpt.logs import logger


import asyncio
import platform
from typing import Any

import fire

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team


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
    return ```tools check:[] ```
    return ```travellingnurse:"" ```
    if you are surgeryassistant,your task is mainly focus on assist in completing the operation and ask for surgical instruments and sterilize them.
    return ```Equipment request:[] ```
    return ```surgeryassistant:"" ```
    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.

    when finished the surgery,stop
    """
    name: str = "Cooperation"

    async def run(self, context: str, name: str, collaborator_name: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, collaborator_name=collaborator_name)
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp


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



async def debate2(idea: str, investment: float = 3.0, n_round: int = 2):
    """Run a team of presidents and watch they quarrel. :)"""
    Surgeryassistant = Tools(name="Surgeryassistant", profile="supply", collaborator_name="Tranvellingnurse")
    Tranvellingnurse = Tools(name="Tranvellingnurse", profile="request", collaborator_name="Surgeryassistant")
    # Solver = Doctor(name="Solver", profile="advice", collaborator_name="Nurse")
    # ReviewDoctor = Doctor(name="ReviewDoctor",profile="review")
    team = Team()
    team.hire([Surgeryassistant, Tranvellingnurse])
    team.invest(investment)

    
    team.run_project(idea, send_to="Surgeryassistant")  
    await team.run(n_round=n_round)



def main(idea: str, investment: float = 3.0, n_round: int =5):

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debate2(idea, investment, n_round),)


if __name__ == "__main__":
    fire.Fire(main) 

