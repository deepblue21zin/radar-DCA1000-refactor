# DBSCAN 핵심 개념
Core point  : 반경 eps 안에 이웃이 min_samples개 이상 있는 점
Border point: 이웃은 부족하지만 core point의 이웃에 포함된 점
Noise       : 어느 클러스터에도 속하지 못한 점

# 초기화
labels  = [-1] * len(features)    # 모든 포인트 → noise로 시작
visited = [False] * len(features) # 아직 아무도 처리 안 됨
cluster_id = 0                    # 첫 클러스터 번호

# 외부 루프 : 모든 포인트 순회

for point_index in range(len(features)):
    if visited[point_index]: continue  # 이미 처리된 포인트는 스킵
    
    visited[point_index] = True
    neighbors = _region_query(features, point_index, eps)
    # eps 반경 안의 모든 이웃 인덱스 반환
    # 자기 자신도 포함됨 (거리=0)
    
    if len(neighbors) < min_samples:
        continue  # core point 아님 → label -1 그대로 (일단 noise)
```

여기서 **visited랑 label은 별개야:**
```
visited = True  → "이 포인트에 대해 region_query를 수행했다"
label = -1      → "아직 어느 클러스터에도 안 들어갔다"

→ visited=True인데 label=-1일 수 있음 (noise 포인트)

# Core 포인트 발견시 클러스터 확장
labels[point_index] = cluster_id  # 이 포인트를 현재 클러스터로 편입
seeds = list(neighbors)           # 탐색할 후보 목록 (큐처럼 동작)
seed_set = set(seeds)             # 중복 방지용 (set은 O(1) 조회)
seed_index = 0
```

`seeds`가 핵심이야. **BFS(너비 우선 탐색)** 처럼 동작해:
```
처음: seeds = [0, 2, 5]  (point_index=1의 이웃들)
         ↓
2 처리 → 2의 이웃 [2, 5, 7, 9] 추가
         ↓
seeds = [0, 2, 5, 7, 9]
         ↓
5 처리 → 5의 이웃 [5, 9, 11] 추가
         ↓
seeds = [0, 2, 5, 7, 9, 11]
...계속 확장

# 내부 while 루프: 클러스터 확장
while seed_index < len(seeds):       # seeds가 동적으로 늘어남
    candidate_index = seeds[seed_index]
    seed_index += 1

- seeds에 계속 추가되면서 리스트가 살아있는 동안 계속 처리해.

if not visited[candidate_index]:
        visited[candidate_index] = True
        candidate_neighbors = _region_query(features, candidate_index, eps)
        
        if len(candidate_neighbors) >= min_samples:  # 이것도 core point면
            for neighbor_index in candidate_neighbors:
                if neighbor_index not in seed_set:   # 새로운 이웃이면
                    seed_set.add(neighbor_index)
                    seeds.append(neighbor_index)     # 탐색 대상에 추가
```

**방문 안 한 포인트만 region_query를 실행**하는 이유:
```
이미 visited=True면 그 포인트의 이웃은 이미 seeds에 추가됐거나
다른 클러스터로 확정된 거야 → 다시 할 필요 없음

if labels[candidate_index] == -1:     # noise였던 포인트면
        labels[candidate_index] = cluster_id  # 이 클러스터로 편입
```

이게 **border point 처리**야:
```
candidate가 visited=True인데 label=-1인 경우
→ region_query를 이미 했는데 이웃이 부족해서 noise였던 포인트
→ 근데 다른 core point의 이웃 반경 안에 들어오면 클러스터에 편입
→ 본인은 core point가 아니지만 클러스터에 속하게 됨 = border point
```

---

## 전체 흐름 시각화
```
포인트: A B C D E  (A,B,C,D는 서로 가깝고 E는 혼자 멀리 있음)
min_samples = 2

1. A 처리 → 이웃=[A,B,C] → core point → cluster_0 시작
   seeds = [A, B, C]

2. B 처리 (seeds에서) → 이웃=[A,B,C,D] → core point
   D가 새로 발견 → seeds = [A, B, C, D]

3. C 처리 → 이웃=[A,B,C] → core point지만 새 이웃 없음

4. D 처리 → 이웃=[B,D] → core point지만 새 이웃 없음

seeds 소진 → cluster_0 완성 {A,B,C,D}
cluster_id = 1

5. E 처리 → 이웃=[E] → 1개 < min_samples=2 → noise
   label[E] = -1 그대로

최종: labels = [0, 0, 0, 0, -1]

