import asyncio
import platform
from typing import Any
import re
import fire
import random
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team


def parse_code(rsp):
    pattern = r"```PR(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text

class Cooperation(Action):
    """Action: Speak out aloud in a debate (quarrel)"""

    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are {name} in a Team of three, you are collaborating on an operation with  {collaboratorA_name},and {collaboratorB_name} based on surgical plan.
    ## SURGERY HISTORY
    Previous rounds:
    {context}
    #1.Correct patient posture
    #2.The operation begins
    #3.The surgeon asked the nurse to pass the suction device to perform neuroendoscopic exploration of the bilateral sphenoid sinus openings.
    #4.Push the right middle turbinate laterally
    #5.The surgeon holds the mirror in his left hand, puts down the monopolar electrosurgery in his right hand, and asks for high-speed grinding of the drill.
    #6.The knife doctor asked the nurse to change the grinding head
    #7.The assistant holds the suction device in his right hand and fills the water with a syringe in his left hand.
    #8.The surgeon put down the grinding drill, it’s time to use the micro hook knife
    #9.The chief surgeon put down the microscopic hook knife and used the suction device instead.
    #10.The surgeon handed the aspirator to the assistant, used tissue collection forceps to collect the specimen and handed it to the nurse.
    #11.The surgeon put down the tissue and used forceps to retain it, and asked the nurse to provide different types of curettes to scrape out the tumor.
    #12.The surgeon puts down the syringe and uses the suction device to explore
    #13.Hand the suction device to the assistant and ask the nurse to provide the artificial dura mater. The instrument nurse asks the circulating nurse to open the artificial dura mater.
    #14.The surgeon uses gun-like forceps to hold Naxi cotton, and the instrument nurse asks the circulating nurse to provide Naxi cotton
    #15.The surgeon, nurse, and anesthetist check the amount of blood loss and the operation record sheet and sign it
    ## YOUR TURN
    Now it's your turn, if you you should closely respond to your {collaboratorA_name} and {collaboratorB_name}'s latest requirement, 
    You need to complete each step of the operation in a conversational manner with other two persons step by step.
    if you are SurgeryDoctor,your task is mainly focus on finish the key surgical steps and Each step needs to describe what you are doing as detailed as possible follow the surgical plan.You will do something wrong if  Probability {p} is greater than 70!!
    return ```surgeon operating:[] ```
    return ```SurgeryDoctor:"" ```
    if you are Nurse, your task is to provide the surgical tools or do some easy surgical steps to help {collaboratorA_name}.
    return ```surgeon operating:[] ```
    return ```Nurse:"" ```
    if you are Solver,your task is mainly focus on analyze the situation of the previous surgical step, Give the possible failure probability of this step and provide a plan for the next surgical step by supervising  {collaboratorA_name} and {collaboratorB_name}' action. If a doctor's error is observed, remedial methods and follow-up procedures will be provided.
    return ```Analyse last step:[] ```
    return ```give next step plan:[] ```
    

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.

    The surgeon in charge holds the mirror in his left hand and the suction device in his right hand, using a monopolar electrosurgical knife


    when finished the surgery,stop
    return ```surgeon operating:[] ```
    """
    name: str = "Cooperation"

    async def run(self, context: str, name: str, collaboratorA_name: str,collaboratorB_name: str,p:int):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, collaboratorA_name=collaboratorA_name,collaboratorB_name=collaboratorB_name,p=random.randrange(70, 101))
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp


class Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    collaboratorB_name: str = ""

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
        todo = self.rc.todo  
        memories = self.get_memories()

        if self.name == "SurgeryDoctor":
            rsp = await todo.run(context=memories, name=self.name, collaboratorA_name=self.collaboratorA_name,collaboratorB_name=self.collaboratorB_name,p=random.randrange(60, 101))
        else:
            rsp = await todo.run(context=memories, name=self.name, collaboratorA_name=self.collaboratorA_name,collaboratorB_name=self.collaboratorB_name,p=random.randrange(1, 50))

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaboratorA_name,
        )
        self.rc.memory.add(msg)

        return msg



async def parallel_world(idea: str, investment: float = 3.0, n_round: int = 20):
    """Run a team of presidents and watch they quarrel. :)"""
    SurgeryDoctor = Doctor(name="SurgeryDoctor", profile="surgery", collaboratorA_name="Nurse",collaboratorB_name="Solver")
    Nurse = Doctor(name="Nurse", profile="assistant", collaboratorA_name="Solver",collaboratorB_name="Solver")
    Solver = Doctor(name="Solver", profile="advice", collaboratorA_name="SurgeryDoctor",collaboratorB_name="Nurse")

    team = Team()
    team.hire([SurgeryDoctor, Nurse,Solver])
    team.invest(investment)

    
    team.run_project(idea, send_to="Solver")  
    await team.run(n_round=n_round)

def main(idea: str, investment: float = 3.0, n_round: int =20):

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(parallel_world(idea, investment, n_round),)


if __name__ == "__main__":
    fire.Fire(main)  

