"""
Filename: MetaGPT/examples/surgon/surgery.py
Created Date: Wednesday, April 24th 2024, 6:52:25 pm
Author: XiaoQi

"""
import asyncio
from datetime import datetime
from metagpt.actions import Action
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
import fire
from metagpt.actions import Action, UserRequirement
from metagpt.schema import Message
import json
from collections import Counter
from difflib import SequenceMatcher
import json
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from doc_actions.actions import *
from doc_roles.roles import *

# 加载JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 构建FAISS索引
def build_index(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    d = X.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(X.toarray())
    return vectorizer, index

# 检索最相似的MRI诊断报告
def search(query, vectorizer, index, texts, top_k=1):
    query_vec = vectorizer.transform([query]).toarray()
    D, I = index.search(query_vec, top_k)
    print(D,I)
    results = [texts[i] for i in I[0]]
    indices = I[0].tolist()
    return results, indices


async def Dialogue1(idea: str, investment: float = 3.0, n_round: int = 2):
    """Run a team of presidents and watch they quarrel. :)"""
    Wardnurse = Nurse(name="Wardnurse", profile="transfer", collaborator_name="Roomnurse")
    Roomnurse = Nurse(name="Roomnurse", profile="receive", collaborator_name="Wardnurse")

    team = Team()
    team.hire([Wardnurse, Roomnurse])
    team.invest(investment)

    
    team.run_project(idea, send_to="Wardnurse") 
    await team.run(n_round=n_round)



async def Dialogue2(idea: str, investment: float = 3.0, n_round: int = 2):

    Anesthesiologist = MazuiRole(name="Anesthesiologist", profile="check", collaborator_name="Roomnurse")
    Roomnurse = MazuiRole(name="Roomnurse", profile="answer", collaborator_name="Anesthesiologist")

    team = Team()
    team.hire([Anesthesiologist, Roomnurse])
    team.invest(investment)
    idea = ""
    
    team.run_project(idea, send_to="Anesthesiologist")  # send debate topic to Biden and let him speak first
    await team.run(n_round=n_round)



async def Dialogue3(idea: str, investment: float = 3.0, n_round: int = 2):
    """Run a team of presidents and watch they quarrel. :)"""
    Surgeryassistant = Tools(name="Surgeryassistant", profile="supply", collaborator_name="Tranvellingnurse")
    Tranvellingnurse = Tools(name="Tranvellingnurse", profile="request", collaborator_name="Surgeryassistant")

    team = Team()
    team.hire([Surgeryassistant, Tranvellingnurse])
    team.invest(investment)

    
    team.run_project(idea, send_to="Surgeryassistant")  
    await team.run(n_round=n_round)


'''
      #1. Proper Patient Posture
      #2. Operation starts
      #3. The surgeon asked the nurse to perform neuroendoscopic exploration of the bilateral sphenoid sinus openings using a suction device.
      #4. (A)Push the right middle turbinate laterally  (B)Push the right middle turbinate longitudinally
      #5. The surgeon holds the mirror with his left hand, puts down the monopolar electric knife with his right hand, and requires the drill bit to be ground at high speed.
      #6. The knife doctor asked the nurse to change the grinding head
      #7. The assistant holds the suction device with his right hand and fills it with water from the syringe in his left hand.
      #8. The surgeon puts down the drill, it's time to use the micro hook knife
      #9. The surgeon put down the microscoil and used the suction device instead.
      #10. The surgeon hands the aspirator to the assistant and uses tissue collection forceps to collect the specimen and hands it to the nurse.
      #11. The surgeon puts down the tissue and holds it with forceps and asks the nurse for different types of curettes to scrape out the tumor.
      #12. The surgeon lowers the syringe and probes using a suction device. (A)tumor could not be completely removed  (B)tumor could  be completely removed
      #13. Use the curette again to scrape out the tumor and do a dance
      #14. Give the suction device to the assistant and ask the nurse to provide the artificial dura mater. (A)find that cerebrospinal fluid leakage  (B) find that no cerebrospinal fluid leakage
      #15. Use a cotton swab to extract leaked cerebrospinal fluid and sing a song
      #15. End of surgery
'''



async def parallel_world(idea: str, investment: float = 3.0, n_round: int = 20):

    SurgeryDoctor = Doctor(name="SurgeryDoctor", profile="surgerydoctor", collaboratorA_name="Nurse",collaboratorB_name="Solver")
    Nurse = Doctor(name="Nurse", profile="nurse", collaboratorA_name="Solver",collaboratorB_name="Solver")
    Solver = Doctor(name="Solver", profile="solver", collaboratorA_name="SurgeryDoctor",collaboratorB_name="Nurse")
    
    team = Team()
    team.hire([SurgeryDoctor, Nurse,Solver])
    team.invest(investment)

    team.run_project(idea, send_to="Solver")  # send debate topic to Biden and let him speak first
    await team.run(n_round=n_round)

    
# idea = """
#         Medical history: The patient is a 56-year-old female. Due to decreased vision in both eyes for 1 year, 
#     she visited the ophthalmology department of many hospitals, but the treatment effect was poor; 6 months ago, 
#     the patient underwent a cranial MRI in another hospital, which showed space-occupying lesions of the tuberculum sellae and meningioma. 
#     possible? The patient was transferred to the neurosurgery department of our hospital for treatment. 
#     Physical examination: clear mind, bilateral pupils of equal sizes and circles, 2.5 mm in diameter, 
#     bilateral pupils with sensitive direct and indirect light reflections, rough visual acuity measurement of both eyes, 
#     counting fingers in meters in the left eye, counting fingers in front of the right eye, Temporal hemianopsia, 
#     no limitation of eye movement, normal limb muscle strength, low muscle tone, physiological reflexes, but no pathological reflexes.

# """

idea = """蝶鞍增大。垂体增大，横径23.0 mm，高15.7 mm，前后径16.9 mm，垂体右翼见较大混杂信号，呈稍短/短T1长T2信号，15.6mm×14.1mm×13.1mm，其内液液平面，增强后边缘可见强化。垂体柄横径3.9 mm，前后径3.5 mm，垂体柄受压左偏。视交叉略受压上抬。双侧海绵窦未见明显异常。垂体后叶短T1信号存在。垂体占位，考虑垂体大腺瘤伴出血可能大。

"""

async def ReviewSurgery(report:str,idea: str, investment: float = 3.0, n_round: int = 1):

    SurgeryReviewer = ReviewDoctor(name="SurgeryReviewer", profile="review",report=report)

    team = Team()
    team.hire([SurgeryReviewer])
    team.invest(investment)   
    team.run_project(idea, send_to="SurgeryReviewer")  
    await team.run(n_round=n_round)

async def execute_surgery(idea:str, investment:int=3.0, n_round:int=5):

    with open('total_en.json', 'r') as file:
        data = json.load(file)    
    num = 1
    for ii,i in enumerate(data[1:131]):
        idea = i["MR"] + i["Preoperative diagnosis"]

        id = i["id"]
        mr = i["MR"]
        diagnoise = i["Preoperative diagnosis"]
        steps = i["Surgical steps"]
        sur_names = i["Surgery name"]

        report = str(i["Surgical steps"])
        with open('static_memories.json', 'r') as files:
            memory = json.load(files)   
        target_phrase =  i["MR"]

        # 将数据转换为DataFrame格式
        df = pd.DataFrame(memory)
        # print(df)
        # 处理缺失值，替换NaN为空字符串
        df['MR'] = df['MR'].fillna('')

        texts = df['MR'].tolist()

        # 构建索引
        vectorizer, index = build_index(texts)

        # 示例查询
        query = target_phrase
        results, indices = search(query, vectorizer, index, texts)
        print(results,indices)

        # 输出对应的手术名称
        for idx in indices:
            # print(f"对应的手术名称: {df.iloc[idx]['手术名称']}")
            sur_name = df.iloc[idx]['surgery_name']
            sur_steps = df.iloc[idx]['standard_Surgical steps']    
            input = "surgery name:" + str(sur_name) + "\nsurgical procedure:" + str(sur_steps)        
            # print(f"对应的手术名称: {df.iloc[idx]['id']}")
        await Plan(input, investment, 1)
        with open('sub_sub_memories.json', 'r') as file:
            data = json.load(file)    
        for i in data:
            record = i
        print("Prepare surgery:")
        await parallel_world(record, investment, n_round)
        print("Surgery completed successfully!")
        # 将列表元素连接为一个字符串
        print("Write a surgical report:")


        # print(idea)
        await ReviewSurgery(report,idea, investment, 1)
        file_path = "sub_memories.json"
        list = []
        with open(file_path, "w") as json_file:
            json.dump(list, json_file)
        with open('static_memories.json', 'r') as f:
            content = json.load(f) 
        new_dict = content[-1]  
        # 添加新的键值对
        new_dict["surgery_name"] = sur_names
        new_dict["id"] = id
        new_dict["MR"] = mr
        new_dict["standard_Preoperative diagnosis"] = diagnoise

        new_dict["standard_Surgical steps"] = steps



        with open('static_memories.json', "w", encoding='utf-8') as f:
            json.dump(content, f,  ensure_ascii=False,indent=2)  

def main(idea:str = "",investment: float = 3.0, n_round: int = 3):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(execute_surgery("idea", investment, n_round),)



if __name__ == "__main__":
    fire.Fire(main)  




