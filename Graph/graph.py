graph = {}
with open('D:\Isa\Kuliah\Semester 2\Analisis Algoritma dan Strukturr Data\Pertemuan 11\data.txt') as f:
    total_vertex, total_edge = map(int, f.readline().split()) # Membaca baris pertama, yaitu vertex dan edge

    # Membaca semua edge
    for _ in range(total_edge):
        # u = vertex awal
        # v = vertex akhir
        # w = edge atau bobot
        u, v, w = map(int, f.readline().split())
        # print(f"u = {u}")
        # print(f"v = {v}")
        # print(f"w = {w}")
        if u not in graph:
            graph[u] = []
            # print(graph[u])
        if v not in graph:
            graph[v] = []
        graph[u].append((v, w))
        graph[v].append((u, w))

    awal = int(f.readline().strip())

def dijkstra(graph, total_vertex, awal):
    jarak = {i: float('inf') for i in range(1, total_vertex + 1)}
    jarak[awal] = 0
    terkunjungi = [False] * (total_vertex + 1)

    while True:
        # Mencari vertex yang belum dikunjungi dengan jarak terkecil
        jarak_terpendek = float('inf')
        vertex_terpendek = -1
        for vertex in range(1, total_vertex + 1):
            # Jika not false == true
            # print(f"jarak terpendek = {jarak_terpendek} : jarak[vertek] = {jarak[vertex]}")
            if not terkunjungi[vertex] and jarak[vertex] < jarak_terpendek: # pertama kali jalan saat vertek ke 981 atau vertex == awal
                # print("jalan")
                jarak_terpendek = jarak[vertex] # pertama ini jadi 0
                vertex_terpendek = vertex # 981
        
        if vertex_terpendek == -1: # Tidak ada vertex yang tersisa untuk diproses
            break

        # Mengunjungi vertex dengan jarak terkecil
        terkunjungi[vertex_terpendek] = True
        # print("-----------------------------------------------------------------")
        # print()
        # print(f"vertex_terpendek :")
        # print(vertex_terpendek)
        # print(graph[vertex_terpendek])
        # print()
        # print("-----------------------------------------------------------------")
        for tetangga, bobot in graph[vertex_terpendek]:
            if jarak[vertex_terpendek] + bobot < jarak[tetangga]:
                # jarak[vertex_terpendek] = jarak[awal] atau 0
                # bobot = 3
                jarak[tetangga] = jarak[vertex_terpendek] + bobot
                # jarak[tetangga] yang tadinya nilainya inf semua menjadi ada 1 yang berniali 3 lalu bisa masuk perulangan for pertama yang membandingkan dengan < inf

    return jarak



jarak = dijkstra(graph, total_vertex, awal)
# for vertex in range(1, total_vertex + 1):
#     print(f"Jarak dari {awal} ke {vertex} adalah {jarak[vertex]}")
# print(jarak.items())

vertex_dengan_jarak_terpanjang = max(jarak, key=jarak.get)
print(f"Simpul dengan jarak terpendek tertinggi dari {awal} adalah {vertex_dengan_jarak_terpanjang} dengan jarak {jarak[vertex_dengan_jarak_terpanjang]}")
print()
print("=======================================")
print()






























# print()
# # print(jarak)
# # print(jarak.items())
# print("-----------------------------------------")
# print(f"Jarak == {jarak}")
# print("-----------------------------------------")
# print()
# print(f"J
# arak.items() == {jarak.items()}")
# print("-----------------------------------------")
# print(jarak)
# print("-------------------------------------------")
# print()
# print("-------------------------------------------")
# print(jarak.items())


# Soal No 2 hanya path

# def dfs_limited(graph, start, target_distance, current_path=None, current_distance=0, depth_limit=1000):
#     if current_path is None:
#         current_path = [start]
#         # print(current_path)

#     if current_distance > target_distance:
#         return None

#     if current_distance == target_distance:
#         return current_path

#     if depth_limit <= 0:
#         return None

#     for neighbor, weight in graph[start]:
#         if neighbor not in current_path:  # memastikan tidak mengunjungi vertex yang sama dua kali
#             new_path = current_path + [neighbor]
#             result = dfs_limited(graph, neighbor, target_distance, new_path, current_distance + weight, depth_limit - 1)
#             if result:
#                 return result

#     return None

# # Mencari jalur dengan panjang 2024 dari vertex awal
# target_distance = 2024
# depth_limit = 1000  # Limit kedalaman DFS
# path = dfs_limited(graph, awal, target_distance, depth_limit=depth_limit)

# if path:
#     print("Ada vertex yang memiliki jarak 2024 dari S.")
#     print("Path:", path)
# else:
#     print("Tidak ada vertex yang memiliki jarak 2024 dari S.")


# print()
# print(f"graph[awal] :")
# print(graph[awal])
# print()
# print(f"graph[524] :")
# print(graph[524])
# print()


# Soal No 2 path dan bobot

def DFS(graph, awal, jarak_target, jalur_sementara=None, jarak_sementara=0): #, depth_limit=1000): #, tes = 0):
    # current_path awalnya berisi None yang nantinya akan berubah berdasarkan new_path
    if jalur_sementara is None: # {1 : Pertama ini jalan, 2 : 2024 is None (false)}
        jalur_sementara = [awal] # [981]

    if jarak_sementara > jarak_target: # {1 : 0 > 2024 (false), 2 : 3 > 2024 (false)}
        return None

    if jarak_sementara == jarak_target: # {1 : 0 == 2024 (false), 2 : 3 == 2024 (false)}
        return jalur_sementara

    #if depth_limit <= 0: # {1 : 1000 <= 0 (false), 2 : 999 <= 0 (false)}
        return None

    # tes = tes+1

    for tetangga, bobot in graph[awal]: # {1 : graph[awal] = graph[981] tetangga dan bobot nya 981, 2 : graph[524]}
        # print(f"neighbor = {tetangga}")
        # print(f"weight = {bobot}")
        # if tes == 2:
            # break

        if tetangga not in jalur_sementara: # {1 : 524 not in [981] (true), 2 : 417 not in [981, 524] (true)}
            jalur_baru = jalur_sementara + [tetangga] # {1 : [981] + [524] = [981, 524], 2 : [981, 524] + [417] = [981, 524, 417]}
            result = DFS(graph, tetangga, jarak_target, jalur_baru, jarak_sementara + bobot) #, depth_limit - 1) #, tes) # Rekursif
                                # graph, 524    , 2024        , [981, 524], 0 + 3, 999
                                # graph, 417    , 2024        ,
            if result:
                return result # jalur_baru

    return None

def menemukan_jalur_dengan_jarak_target(graph, awal, jarak_target):
    jalur = DFS(graph, awal, jarak_target,) # path = [981, 524, 417, ...]
    if jalur:
        return jalur
    return None

def mendapatkan_jalur_dengan_bobotnya(graph, jalur):
    bobot_jalur = []
    # print(len(path))
    for i in range(len(jalur) - 1):
        for tetangga, bobot in graph[jalur[i]]:
            if tetangga == jalur[i + 1]: 
                bobot_jalur.append((jalur[i], tetangga, bobot))
                break
    return bobot_jalur


jarak_target = 2024
jalur = menemukan_jalur_dengan_jarak_target(graph, awal, jarak_target)

if jalur:
    print("Ada vertex yang memiliki jarak 2024 dari S.")
    bobot_jalur = mendapatkan_jalur_dengan_bobotnya(graph, jalur)
    # print("Path:", " -> ".join(str(node) for node in path))
    # print("Dengan bobot:")
    N = 0
    for u, v, w in bobot_jalur:
        print(f"{u} -> {v} dengan bobot {w}")
        N = N + w
        print(f"Total bobot = {N}")
        
else:
    print("Tidak ada vertex yang memiliki jarak 2024 dari S.")

print()
print("=======================================")
print()






















def read_graph_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    N, M = map(int, lines[0].strip().split())
    
    edges = []
    for line in lines[1:M+1]:
        u, v, _ = map(int, line.strip().split())
        edges.append((u, v))
    
    S = int(lines[M+1].strip())
    
    return N, M, edges, S


filename = './Pertemuan 11/data.txt'
N, M, edges, S = read_graph_from_file(filename)





def bfs_farthest_vertices(N, edges, S):
    graph = {i: [] for i in range(1, N + 1)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    queue = [S]
    distances = {S: 0}
    max_distance = 0
    farthest_paths = []

    while queue:
        vertex = queue.pop(0)
        current_distance = distances[vertex]
        
        for neighbor in graph[vertex]:
            if neighbor not in distances:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
                if distances[neighbor] > max_distance:
                    max_distance = distances[neighbor]
                    farthest_paths = [[S] + [neighbor]]
                elif distances[neighbor] == max_distance:
                    farthest_paths.append([S] + [neighbor])
    
    return max_distance, farthest_paths

max_distance_bfs, farthest_paths_bfs = bfs_farthest_vertices(N, edges, S)

print(f"BFS Maximum distance from {S}: {max_distance_bfs}")
for path in farthest_paths_bfs:
    print(f"BFS Path: {path}")


print()
print("==============================================")
print()


















def dfs_farthest_vertices(graph, start):
    stack = [(start, [start])]
    visited = set()
    max_distance = 0
    farthest_paths = []

    while stack:
        vertex, path = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)

        if len(path) - 1 > max_distance:  
            max_distance = len(path) - 1
            farthest_paths = [path]
        elif len(path) - 1 == max_distance:
            farthest_paths.append(path)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return max_distance, farthest_paths

graph = {i: [] for i in range(1, N + 1)}
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

max_distance_dfs, farthest_paths_dfs = dfs_farthest_vertices(graph, awal)

print(f"DFS Maximum distance from {awal}: {max_distance_dfs}")
for path in farthest_paths_dfs:
    print(f"DFS Path: {path}")
print()
print("=======================================")
print()









































# Worst case no 3

# import sys
# sys.setrecursionlimit(9999)

# longest_path_found = []

# def DFS(graph, node, visited=None, path=None, longest_path=None):
#     if visited is None:
#         visited = set()
#     if path is None:
#         path = [node]
#     if longest_path is None:
#         longest_path = path

#     visited.add(node)

#     global longest_path_found
#     if len(longest_path_found) >= len(graph.keys()):
#         return longest_path_found
#     elif len(path) > len(longest_path_found):
#         longest_path_found = path

#     for neighbor in graph[node]:
#         if neighbor not in visited:
#             new_path = path + [neighbor]
#             if len(new_path) > len(longest_path):
#                 longest_path = new_path
#             longest_path = DFS(graph, neighbor, visited, new_path, longest_path)

#     visited.remove(node)
#     return longest_path


# # Define your graph here. For example:
# graph = {}
# f = open("Pertemuan 11/data.txt", "r")
# nv, ne = map(int, f.readline().split())

# for i in range(ne):
#     src, dest, _ = map(int, f.readline().split())
#     if src not in graph:
#         graph[src] = []
#     if dest not in graph:
#         graph[dest] = []
#     graph[src].append(dest)
#     graph[dest].append(src)

# # Specify your source node
# source_node = int(f.readline())  # Replace '1' with your source node

# # Find the longest path using DFS
# longest_path = DFS(graph, source_node)

# print(
#     f"The longest path from source node {source_node} is {longest_path[-1]} with length {len(longest_path)-1}"
# )