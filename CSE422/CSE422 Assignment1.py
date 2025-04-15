import heapq
from pathlib import Path

home = Path.home()/"Documents/University/Spring '25/CSE422"
headDict, nodeDict = {}, {}

with open(home/"Input file.txt", encoding = "utf-8") as fileIn:
    inputFile = fileIn.readlines()

for i in inputFile:
    row = i.split()
    headDict[row[0]] = int(row[1])
    box = []

    for j in range(2, len(row), 2):
        box.append((row[j], int(row[j + 1])))
    nodeDict[row[0]] = box

def aStarSearch(start, end, headDict, nodeDict):
    visited = set()
    store = []

    heapq.heappush(store, (headDict[start], 0, start, [start]))

    while store:
        s, goal, point, path = heapq.heappop(store)
        if point in visited:
            continue
        visited.add(point)
        if point == end:
            return path, goal

        for key, value in nodeDict[point]:
            if key not in visited:
                G = goal + value
                F = G + headDict[key]
                heapq.heappush(store, (F, G, key, path + [key]))
    return None

aStar = aStarSearch(input("Start node: "), input("Destination: "), headDict, nodeDict)

print("Path: "," -> ".join(aStar[0]))
print(f"Total Cost: {aStar[1]}")
