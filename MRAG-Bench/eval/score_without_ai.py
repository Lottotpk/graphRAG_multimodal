import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default=None, help='File to be evaluated')
    args = parser.parse_args()

    data = {}
    total_qs = 0
    total_correct = 0
    with open(args.filename, 'r') as f:
        for line in f:
            total_qs += 1
            choice = json.loads(line)
            if choice['scenario'] not in data:
                data[choice['scenario']] = [0, 1]
            else:
                data[choice['scenario']][1] += 1
            
            if choice['output'][0] == choice['gt_choice']:
                data[choice['scenario']][0] += 1
                total_correct += 1
    
    print("--------------Results--------------")
    for key, value in data.items():
        print(f"{key}: {round(100*value[0]/value[1], 3)}%")
    print(f"Total accuracy: {round(100*total_correct/total_qs, 3)}%")
    print("-----------------------------------")