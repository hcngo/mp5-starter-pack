from pyspark import *
from pyspark.sql import SparkSession
from graphframes import *

sc = SparkContext()
spark = SparkSession.builder.appName('fun').getOrCreate()


def get_connected_components(graphframe):
    # TODO:
    # get_connected_components is given a graphframe that represents the given graph
    # It returns a list of lists of ids.
    # For example, [[a_1, a_2, ...], [b_1, b_2, ...], ...]
    # then a_1, a_2, ..., a_n lie in the same component,
    # b_1, b2, ..., b_m lie in the same component, etc
    # return [[]]
    components_map = {}
    results_df = graphframe.connectedComponents()
    results = results_df.rdd.collect()
    for v in results:
        component = v["component"]
        if not (component in components_map):
            components_map[component] = []
        components_map[component].append(v["id"])
    return list(components_map.values())

if __name__ == "__main__":
    vertex_list = []
    edge_list = []
    with open('dataset/graph.data') as f:  # Do not modify
        for line in f:
            vertices = line.split(" ")
            src = vertices[0]
            dst_list = [v.strip() for v in vertices[1:]]
            vertex_list.append((src,))
            edge_list += [(src, dst) for dst in dst_list]

    vertices = spark.createDataFrame(
        vertex_list,
        ["id"]
    ) 
    edges = spark.createDataFrame(
        edge_list,
        ["src", "dst"]
    )

    g = GraphFrame(vertices, edges)
    sc.setCheckpointDir("/tmp/connected-components")

    result = get_connected_components(g)
    for line in result:
        print(' '.join(line))
