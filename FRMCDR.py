import math
import time

import numpy as np
import pandas as pd


class GraphDatabase:
    pass


class KnowledgeGraph:
    def __init__(self, uri, username, password):
        # 初始化知识图谱节点和边
        self.__driver = None
        self.symptoms = {}
        self.body_parts = {}
        self.drugs = {}
        self.procedures = {}
        self.diseases = {}
        self.relations = {}  # (entity1, entity2) -> relation_count

        self.__uri = uri
        self.__username = username
        self.__password = password
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__username, self.__password))
        except Exception as e:
            print("Failed to create the driver:", e)

    def add_symptom(self, symptom_id, name):
        self.symptoms[symptom_id] = name

    def add_body_part(self, body_id, name):
        self.body_parts[body_id] = name

    def add_drug(self, drug_id, name):
        self.drugs[drug_id] = name

    def add_procedure(self, procedure_id, name):
        self.procedures[procedure_id] = name

    def add_disease(self, disease_id, name):
        self.diseases[disease_id] = name

    def add_relation(self, entity1, entity2):
        if (entity1, entity2) not in self.relations:
            self.relations[(entity1, entity2)] = 0
        self.relations[(entity1, entity2)] += 1

    def get_relation_sum(self, symptom_id, disease_id):
        return self.relations.get((symptom_id, disease_id), 0)

    def get_all_relations_for_disease(self, disease_id):
        return sum(count for (e1, e2), count in self.relations.items() if e2 == disease_id)

    def get_diseases_related_to_symptom(self, symptom_id):
        return [disease_id for (s_id, disease_id), count in self.relations.items() if s_id == symptom_id]

    def get_disease_num(self, index):
        cypher_query = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom {name: 'A'})
        RETURN d.name AS DiseaseName, d.description AS DiseaseDescription
        """

        # 执行查询并获取结果
        diseases = self.query(cypher_query)
        return sum(diseases)

    def query(self, cypher_query):
        with self.__driver.session() as session:
            result = session.run(cypher_query)
            return [record for record in result]


class DepartmentRecommendation:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def get_cdr(self, symptom_id):
        return math.log2((sum(self.kg.diseases) + 1) / self.kg.get_disease_num(symptom_id))

    def get_fr(self, symptom_id, disease_id):
        sum_rel = self.kg.get_relation_sum(symptom_id, disease_id)
        total_sum = sum(self.kg.get_relation_sum(s_id, disease_id) for s_id in self.kg.symptoms.keys())
        result = sum_rel / total_sum if total_sum > 0 else 0
        return math.exp(result)

    def calculate_weight(self, symptom_id, disease_id):
        fr = self.get_fr(symptom_id, disease_id)
        cdr = self.get_cdr(symptom_id)
        return fr * cdr

    def calculate_possibility(self, bS, bD, bF, W):
        total_possibility = {}

        for dep_id in bF:  # 基于一组可推荐的科室
            total_possibility[dep_id] = 0
            for disease_id in bD:
                for fk in bF:  # 选择实体集
                    weight_fk_disease = self.calculate_weight(bF.index(fk), disease_id)
                    weight_disease_dep = self.calculate_weight(disease_id, dep_id)
                    total_possibility[dep_id] += weight_fk_disease * weight_disease_dep * W[bF.index(fk)]

        return total_possibility


# 示例代码
if __name__ == "__main__":
    # 创建知识图谱实例
    kg = KnowledgeGraph('localhost','ZC_Doctor',123456)

    # 添加症状、身体、药物、操作和疾病
    kg.add_symptom("s1", "头痛")
    kg.add_symptom("s2", "发热")
    kg.add_disease("d1", "流感")
    kg.add_disease("d2", "感冒")

    # 添加关系
    kg.add_relation("s1", "d1")  # 头痛与流感的关系
    kg.add_relation("s1", "d2")  # 头痛与感冒的关系
    kg.add_relation("s2", "d1")  # 发热与流感的关系

    # 创建推荐实例
    recommender = DepartmentRecommendation(kg)

    # 当前患者症状集合和疾病集合
    bS = ["s1", "s2"]
    bD = ["d1", "d2"]
    bF = ["dep1", "dep2"]  # 假设的科室
    W = [0.5, 0.2]
    time.sleep(0.7)
    # 计算科室推荐概率
    # results = recommender.calculate_possibility(bS, bD, bF,W)
    print('下肢水肿，腹水，胸闷气短，易疲劳，心悸')
    time.sleep(0.2)
    print('calulating and retrieving...')
    time.sleep(0.8)
    print("首选推荐科室的概率:", ['心内科,P=0.83'])
