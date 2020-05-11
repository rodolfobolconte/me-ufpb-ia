#Resultados Árvore de Decisão ID3 (Iterative Dichotomiser 3):
#Acurácia: 0.8975
#Precisão: 0.8943
#Sensibilidade: 0.8962

import pandas as pd
from sklearn.model_selection import train_test_split
from id3 import Id3Estimator
from id3 import export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

dados = pd.read_csv('amostras.csv', sep=',', encoding='utf8')

dadosX = dados[['pontuacao_final','sobrevivencia','bonus_ultima_sobrevivencia','dano_disparo','bonus_disparo_morte','colisao_dano','bonus_colisao_morte','1lugar','2lugar','3lugar']].values
dadosY = dados['classificacao']

treinoX, testeX, treinoY, testeY = train_test_split(dadosX, dadosY, test_size=0.3, shuffle=False)

modeloArvodeID3 = Id3Estimator(max_depth=3)

modeloArvodeID3.fit(treinoX, treinoY)

export_graphviz(modeloArvodeID3.tree_, 'arvoreExecutada.dot', ['pontuacao_final','sobrevivencia','bonus_ultima_sobrevivencia','dano_disparo','bonus_disparo_morte','colisao_dano','bonus_colisao_morte','1lugar','2lugar','3lugar'])

classificacoes = modeloArvodeID3.predict(testeX)

print('Resultados Árvore de Decisão ID3 (Iterative Dichotomiser 3):')
print('Acurácia: %.4f' %accuracy_score(classificacoes, testeY))
print('Precisão: %.4f' %precision_score(classificacoes, testeY, average='macro'))
print('Sensibilidade: %.4f' %recall_score(classificacoes, testeY, average='macro'))