# Computer Assignment #1
Consider the simple two cost functions:
 $$J_1=(x_1^2+ x_2^2 - x_1x_2-10)10^{-3} $$
 $$J_2=x_1^2 +(x_2-2)^2-100$$

 Write your own program to minimize each function using Genrtic Algorithm.
<pre>
1. Encoding of population   
2. Selection of fitness function  [환경에 대한 적응도를 나타내는 것, 문제 맞게 변형해서 반영]
3. Evaluation of Fitness function 
4. Parents Selection  [영향도가 큰 그룹만 다음세대에 반영 가능]
5. Crossover process [다음 세대를 만든다]
6. Mutatio Process [돌연변이]

구현방식
  Parents Selection 
  => roulette_wheel_selection 
  Crossover process 
  => uniform_crossover
</pre>




    