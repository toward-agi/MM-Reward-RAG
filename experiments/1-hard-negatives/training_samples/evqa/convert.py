import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # List to store the converted data
    data = []
    
    # Read the CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Convert each row to a dictionary and add to data list
        for row in csv_reader:
            data.append(row)
    
    # Write to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    return data

# Example usage
if __name__ == "__main__":
    input_file = "/data/mingyan/encyclopedic/train.csv"
    output_file = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/training_samples/evqa/train.json"
    
    result = csv_to_json(input_file, output_file)

    url2entityid = json.load(open("/data/mingyan/encyclopedic/wiki_url_to_entity.json", "r"))
    i = 0

    filtered_result = []

    for item in result:
        if item["wikipedia_url"] in url2entityid:
            item["entity_id"] = url2entityid[item["wikipedia_url"]]
            filtered_result.append(item)
        else:
            i = i + 1
    print(f"Total records converted: {len(result)}")
    print(f"Total records with missing entity_id: {i}")
    with open(output_file, "w") as f:
        json.dump(filtered_result, f, indent=4)
    print(f"Filtered records saved to {output_file}")
