# Avaliação gráfica da CNN em R

O script usa apenas R base, lê os resultados existentes e gera sete figuras em PNG e SVG, tabelas de apoio e uma galeria HTML. Não é necessário treinar os modelos novamente ou instalar pacotes.

## Executar no RStudio

```r
source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/06_avaliacao_grafica.R", encoding = "UTF-8")
gerar_graficos_cnn()
```

Os resultados são gravados em `D:/usuario_armazenamento/cassio/R/deep_learning_caret/outputs/avaliacao_grafica`. Abra `galeria.html` nessa pasta.

Para escolher outro destino ou outra execução final:

```r
gerar_graficos_cnn(
  output_dir = "D:/usuario_armazenamento/cassio/R/deep_learning_caret/outputs/avaliacao_grafica",
  final_run_id = "final_20260823_001856",
  config_id = "cfg_014"
)
```

O script foi preparado para a estrutura de resultados de `soc_stock_0_5cm`. O identificador do tuning é obtido do resumo da execução final. Os arquivos originais do projeto são necessários; o pacote de figuras sozinho não contém as predições de entrada. O cálculo de todas as combinações pode levar alguns minutos.

## Figuras

1. Ranking das configurações na validação.
2. Desempenho, tempo de treinamento e tamanho da arquitetura.
3. Curvas de aprendizado das sementes.
4. Estabilidade na validação e comparação com o tuning original.
5. Observado versus predito e resíduos do ensemble no teste.
6. Ganho ao combinar sementes, considerando todos os subconjuntos.
7. Magnitude e direção do erro por faixa de estoque observado.

O ensemble usa a mediana das predições em escala nativa. A faixa entre combinações não é intervalo de confiança. O orçamento de treinamento também muda entre tuning e retreinamento; essa comparação não isola o efeito da semente. Os gráficos do teste são descritivos e não devem ser usados para selecionar sementes.

O número de parâmetros é calculado analiticamente segundo a arquitetura atual do projeto; alterações na arquitetura exigem atualizar a função de contagem. As figuras são recriadas no destino a cada execução.
