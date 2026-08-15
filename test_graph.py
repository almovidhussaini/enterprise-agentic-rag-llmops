from app.agents.graph import graph
from pprint import pprint


result = graph.invoke(

    {

        "question":"Explain encoder discussed in the paper",
        # "question":"who invented python",

        "route":"",

        "retrieved_docs":[],

        "answer":""

    }

)

print(result["route"],"route")
print(result["answer"],"answer")