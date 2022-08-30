## 2つのベクトルのなす角
 - 内積とベクトル大きさを使って余弦を求め，その後逆三角関数で角度を求めます

  $$  cos \theta = {\vec{v_1} \cdot \vec{v_2} \over |\vec{v_1}||\vec{v_2}|} $$

   - ベクトルの大きさは numpy.linalg.norm 関数で，内積は numpy.inner 関数で求めることができます
   - 逆コサインには numpy.arccos 関数，角速度から角度への変換には numpy.rad2deg 関数を利用できます
      -- 逆三角関数は mathパッケージにも実装されています（acos，asin，atan）
  
  import numpy as np

def calcAngle(v1, v2):
  v1_n = np.linalg.norm(v1)
  v2_n = np.linalg.norm(v2)

  cos_theta = np.inner(v1, v2) / (v1_n * v2_n)

  return np.rad2deg(np.arccos(cos_theta))

def main():
  v1 = np.array([1, 1, 1])
  v2 = np.array([1, 1, 0])

  print(calcAngle(v1, v2))  

  v1 = np.array([3, 1])
  v2 = np.array([4, 5])

  print(calcAngle(v1, v2))  


if __name__ == '__main__':
  main()
  
