import json

file_path = "../data/semantic_layer.json"

with open(file_path, 'r') as f:
    semantic_layer = json.load(f)

# print(semantic_layer.keys())
# print(semantic_layer["tables"].keys())
# make_json_tables = {}
make_json_metrics = {}


# for table_name, table_info in semantic_layer["tables"].items():
#     make_json_tables[table_name] = {
#         "description": table_info.get("description", "No description available"),
#         "synonyms": table_info.get("synonyms", []),
#         "business_context": table_info.get("business_context", "No business context available")
#     }

# print(make_json_tables)

for metric_name, metric_info in semantic_layer["metrics"].items():
    make_json_metrics[metric_name] = {
        "description": metric_info.get("description", "No description available"),
        "synonyms": metric_info.get("synonyms", [])
    }

print(make_json_metrics)
# def prepare_semantic_context_for_router(file_path):
#     with open(file_path, 'r') as f:
#         semantic_layer = json.load(f)
    
#     list_of_tables = semantic_layer.get("tables", [])
#     list_of_metrics = semantic_layer.get("metrics", [])
#     list_of_join_paths = semantic_layer.get("join_paths", [])
    
#     return list_of_tables, list_of_metrics, list_of_join_paths