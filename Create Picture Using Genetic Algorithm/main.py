import cv2, random, os, sys
import numpy as np
from copy import deepcopy
from skimage.measure import compare_mse
import multiprocessing as mp

img = cv2.imread('test.png')
height, width, channels = img.shape

#パラメータ
initial_Genes_Number = 50
population_Number = 50
probability_Mutation = 0.01
probability_Add = 0.3
probability_Remove = 0.2


min_Radius, max_Radius = 5, 15
gene_Count_Check = 100

#region 遺伝子 Class
class Gene():
  def __init__(self):#遺伝子の性質
    self.radius = random.randint(min_Radius, max_Radius)#円の大きさ
    self.center = np.array([random.randint(0, width), random.randint(0, height)])#円の位置
    self.color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])#円の色

  #region 突然変異を起こるMethod
  def Mutate(self):
    mutation_Size = max(1, int(round(random.gauss(15, 4)))) / 100 #最低が0.01、 平均が0.15のmutation size、結局 0.85と 1.25の 変化が生じる。
    random_Probability = random.uniform(0, 1)
    if random_Probability < 0.33: #円の半径
      self.radius = np.clip(random.randint(
        int(self.radius * (1 - mutation_Size)),
        int(self.radius * (1 + mutation_Size))),
        1, 100)
    elif random_Probability < 0.66: #円の位置
      self.center = np.array([
        np.clip(random.randint(
          int(self.center[0] * (1 - mutation_Size)),
          int(self.center[0] * (1 + mutation_Size))),
          0, width)
        ,#X
        np.clip(random.randint(
          int(self.center[1] * (1 - mutation_Size)),
          int(self.center[1] * (1 + mutation_Size))),
          0, height)
         #Y
      ])
    else: #円の色
      self.color = np.array([
        np.clip(random.randint(int(self.color[0] * (1 - mutation_Size)),int(self.color[0] * (1 + mutation_Size)))
          ,0, 255) #clip varation
        ,#Blue
        np.clip(random.randint(
          int(self.color[1] * (1 - mutation_Size)),
          int(self.color[1] * (1 + mutation_Size)))
          ,0, 255)#clip varation
        ,#Green
        np.clip(random.randint(
          int(self.color[2] * (1 - mutation_Size)),
          int(self.color[2] * (1 + mutation_Size))),
          0, 255)
         #Red
      ])
  #endregion

#region 遺伝子の世代繁殖 Method
def Compute_Population(g):
  genome = deepcopy(g)

  for gene in random.sample(genome, k=int(len(genome) * probability_Mutation)): #世代の突然変異
      gene.Mutate()

  #遺伝子の再生
  if random.uniform(0, 1) < probability_Add:
    genome.append(Gene())

  #遺伝子の除去
  if len(genome) > 0 and random.uniform(0, 1) < probability_Remove:
    genome.remove(random.choice(genome))

  new_fitness, new_out = Visualize_Genome_Compute_Fitness(genome) #

  return new_fitness, genome, new_out
#endregion


#region 遺伝子を絵描いてFitness計算
def Visualize_Genome_Compute_Fitness(genome):
  out = np.ones((height, width, channels), dtype=np.uint8) * 255 #白画面

  for gene in genome:
    cv2.circle(out, center=(gene.center[0],gene.center[1]), radius=gene.radius, color=(int(gene.color[0]), int(gene.color[1]), int(gene.color[2])), thickness=-1)
  # MSEを検査
  fitness = 255 / compare_mse(img, out)
  return fitness, out
#endregion

#region MAIN
if __name__ == '__main__':
  os.makedirs('result', exist_ok=True) #countごとに保存するフォルダを作成
  p = mp.Pool(mp.cpu_count() - 1) #並列処理

  # 第一遺伝子
  best_Genome = [Gene() for gene in range(initial_Genes_Number)]
  best_Fitness, best_out = Visualize_Genome_Compute_Fitness(best_Genome)

  genome_Number = 0

  while True:
    try:
      results = p.map(Compute_Population, [deepcopy(best_Genome)] * 50)
    except KeyboardInterrupt:
      p.close()
      break

    results.append([best_Fitness, best_Genome, best_out])
    new_fitnesses, new_genomes, new_outs = zip(*results) #deepCopy
    best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda temp_results: temp_results[0], reverse=True)

    best_Fitness, best_Genome, best_out = best_result[0] #fitnessが最高であることに合わせてgenomeとoutを整列
    print('Generation #%s, Fitness %s' % (genome_Number, best_Fitness))#世代の結果
    genome_Number += 1

    if genome_Number % gene_Count_Check == 0:# Check イメージ再生
      cv2.imwrite('result/%s.jpg' % genome_Number, best_out)

    cv2.imshow('best out', best_out)# 現在世代のイメージ
    if cv2.waitKey(1) == ord('q'):
     p.close()
     break

  cv2.imshow('best out', best_out)
  cv2.waitKey(0)
#endregion


