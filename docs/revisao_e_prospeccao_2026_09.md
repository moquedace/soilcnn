# Revisão de concepção e prospecção — setembro/2026

Auditoria do projeto desde a concepção e prospecção de rumos. **Nada aqui foi
executado** — é material para decidir o que fazer, com o porquê, o custo e o
risco de cada caminho.

Escrito enquanto o `03` rodava (3 configs × 3 folds × 3 sementes, espacial).

---

## Parte 0 — Resumo executivo

**O que está sólido:** a camada `R/` não sabe nada do domínio do exemplo (zero
menções a SOC/solo/WOSIS, zero caminhos absolutos), 229 asserções em 9 arquivos
de teste, e os três incidentes sérios de 2026-09 (corrupção do `torch_save`,
vazamento espacial, tipo adivinhado na releitura) viraram teste em vez de
lembrança.

**O achado que mais dói:** a regra de buffer que eu escrevi ontem **está
errada por geometria**, e a verificação independente desta madrugada pegou.
Sobreposição de patch é uma condição de *quadrado*; o buffer mede *círculo*.
Com `buffer = 15 × res`, 0,51% a 0,98% da validação por fold ainda dividia
pixels de patch com o treino — enquanto o relatório mostrava um limpo 0% de
"mesmo pixel". Detalhe em [§A0](#a0-o-buffer-está-medindo-o-círculo-errado).

**O achado mais sério de método:** o **conjunto de teste continua sendo um
split aleatório**. A etapa 4 tornou a *validação* espacialmente limpa (0% por fold),
mas 28,08% dos pontos de teste ainda dividem pixel de 250 m com um ponto de
treino. Toda métrica `test_*` que este pipeline produz hoje é otimista, e nada
no relatório diz isso no momento em que ela é lida.

**A decisão mais urgente, porque expira:** `padding = "valid"` nas convoluções
([§B1](#b1-predição-totalmente-convolucional-exige-uma-decisão-de-arquitetura-agora)).
É o que permite a predição ficar ~100× mais rápida, e **só vale se decidido
antes do treino final** — depois, significa retreinar tudo.

**O que falta para ser pacote:** um registro de modelos. Hoje
`build_cnn_from_config()` conhece uma arquitetura só. O que faz o caret ser
caret é `method = "rf"` vs `method = "svmRadial"` ([§C1](#c1-registro-de-modelos)).

---

## Parte 1 — Diagnóstico: o que está genuinamente bom

Não é elogio: são propriedades verificáveis, com a evidência ao lado.

| propriedade | evidência |
|---|---|
| Framework desacoplado do domínio | `grep -i "soc\|soil\|wosis\|ton_ha" R/*.R` → **0 linhas** |
| Sem caminho absoluto na camada de framework | `grep "D:/\|C:/" R/*.R` → **0 linhas** |
| Falhas viraram teste, não lembrança | 229 asserções; `test_patch_store_io` existe porque o `torch_save` corrompeu; `test_resample` porque bloco sem buffer não separa |
| Geometria de patch unificada | `R/patches.R`: um só caminho de indexação, provado contra 6.642.157 células reais com 0 divergências |
| QC separado de escalonamento | é o que permite escalonar por fold sem reextrair 16 GB — provado em `test_preprocess` |
| Decisões documentadas com alternativas descartadas | `docs/design_decisions.md` (14 seções) + `docs/project_log.md` (982 linhas) |
| Verificação em duas camadas | `run_all` (o código está certo, 3 s) + `99_check_pipeline` (este run está certo, dados reais) |

O ponto que eu destacaria: **o `99` distingue "não iniciado" de "falhou"**, e o
snapshot responde "mudou alguma coisa?" por diff em vez de rolagem de tela.
Isso é raro em pipeline de pesquisa e é o que tornou possível fazer um refactor
desse tamanho sem perder o chão.

---

## Parte 2 — O que está fraco, em ordem de gravidade

### 2.1 — O conjunto de teste é um split aleatório (ALTO)

O `99` mede, todo run:

```
test | same raster cell        | 28,08%  (1.561 de 5.560)
test | patches overlap (15x15) | 55,0%
```

A etapa 4 resolveu a validação — `spatial_folds` + buffer dão 0% por fold. Mas
o teste foi separado lá no `01` por amostragem estratificada pelo alvo, e nunca
foi tocado desde então. **Mais da metade dos pontos de teste compartilha pixels
de patch com pontos de treino.**

**Importante não exagerar o achado.** Patches vizinhos compartilharem pixels
**não é defeito** — é o que amostras vizinhas são, e num split aleatório é
exatamente a condição sendo medida. O que é defeito, em qualquer split, é a
**mesma célula**: dois perfis no mesmo pixel de 250 m têm entrada idêntica bit a
bit, e um pode ser avaliado no que o outro treinou.

Então o problema aqui não é o teste ser aleatório: é os dois números
conviverem na mesma tabela respondendo a **perguntas diferentes** —
interpolação perto de amostra conhecida, e predição em terreno não visitado —
com nomes que sugerem que o segundo é o mais rigoroso.

**Não é erro de código** — é uma decisão que ficou para trás quando a validação
evoluiu. Ver [§A1](#a1-teste-espacialmente-independente).

### 2.2 — As três funções sem teste são as três que mentem em silêncio (ALTO)

Levantei quais funções de `R/` nenhum teste exercita diretamente. O padrão é
desconfortável:

| função | o que faz | modo de falha |
|---|---|---|
| `spatial_overlap_report()` | mede vazamento entre splits | **sub-reporta → 0% falso** |
| `calc_metrics()` | CCC, NSE, RPD, MQI | **erro de fórmula muda todo ranking** |
| `write/compare_run_snapshot()` | detecta mudança entre runs | **"tudo idêntico" falso** |

As três compartilham a mesma característica: quando quebram, elas dizem que
está tudo bem. É a categoria que mais precisa de teste e é a que não tem.
`calc_metrics` é particularmente barato de cobrir — CCC e NSE têm fórmula
fechada, dá para conferir contra valor calculado à mão em 20 linhas.

Agravante: desde ontem o pipeline **afirma** "0% de vazamento por fold" com
base em `spatial_overlap_report`, e essa afirmação nunca foi verificada contra
um caso onde a resposta é conhecida.

### 2.3 — O teste é avaliado a cada config durante o tuning (MÉDIO-ALTO)

`train_one_cnn()` roda `predict_loader()` sobre o teste ao final de **toda**
unidade, e escreve `test_ccc`/`test_mae` na tabela de comparação. Com 27
unidades, o conjunto de teste é olhado 27 vezes antes de qualquer decisão.

O código está correto — a seleção usa só validação, e isso está documentado
(`design_decisions.md` §12). Mas a informação está na tela, e quem decide é
humano. Um conjunto de teste consultado 27 vezes durante a escolha não é mais
um conjunto de teste.

**Correção barata:** `evaluate_test = FALSE` como default em `run_cnn_tuning()`,
`TRUE` só no `04`. Custa uma linha e devolve tempo de CPU de brinde.

### 2.4 — A validação faz dois trabalhos ao mesmo tempo (MÉDIO)

O mesmo conjunto de validação (a) para o treino via early stopping e (b)
ranqueia as configs. A época escolhida é a melhor *naquele* conjunto, e a
métrica reportada é medida *nesse mesmo* conjunto — é um máximo, não uma
estimativa não-viesada.

Isso infla `val_ccc` de forma sistemática, e o viés cresce com o número de
épocas e de configs. É o argumento clássico para *nested resampling*
([§A3](#a3-separar-early-stopping-de-seleção)).

### 2.5 — O README descreve o pipeline anterior ao refactor (MÉDIO)

> `02_extract_patches.R` — Spatial patch arrays N × C × H × W (**scaled**, channel-aligned)

Os patches são gravados **crus** desde o refactor; o escalonamento passou a ser
por fold. O README é a primeira coisa que um terceiro lê, e ele descreve um
desenho que não existe mais — incluindo o que era exatamente o defeito
corrigido (patches presos a um split).

### 2.6 — Cópias divergentes do núcleo (MÉDIO)

```
_test_gpu/R/cnn_architecture.R   367 linhas  vs  R/  402   DIVERGIU
_test_gpu/R/train_cnn.R          601 linhas  vs  R/  988   DIVERGIU
_test_gpu/R/utils.R              195 linhas  vs  R/  280   DIVERGIU
scripts/01,02,03_*.R           3.197 linhas  — pipeline pré-refactor inteiro
```

São armadilhas: alguém (inclusive você daqui a seis meses) abre o arquivo
errado, ou pior — edita o errado e não entende por que nada mudou.

### 2.7 — Idioma misto dentro do framework (MÉDIO, para pacote)

`R/` tem ~1.200 linhas de comentário em inglês e **47 em português** — todas
recentes, introduzidas por mim. Pior: `print_noise_floor()` imprime em
português. É saída da camada de framework, que um terceiro vai ver.

Decisão a tomar, não ambiguidade a manter: framework inteiro em inglês
(alcance) ou inteiro em português (seu público imediato). Os exemplos e o log
podem ficar em português em qualquer dos casos.

### 2.8 — Itens menores, todos com correção conhecida

- **`DescTools` por uma fórmula de 4 linhas.** `calc_metrics` carrega uma
  dependência pesada para calcular CCC — que ainda computa intervalo de
  confiança que ninguém usa, **a cada época**, sobre 10 mil pontos. Implementar
  direto remove a dependência e acelera o laço de treino.
- **MQI como coluna de primeira classe.** Métrica inventada aqui (documentado
  honestamente em `design_decisions.md` §13). Num pacote, deveria ser opt-in:
  um usuário não deve receber uma coluna sem precedente na literatura sem pedir.
- **A cabeça da rede é fixa** (`embedding → 128 → 64 → 1`), fora do `tune_grid`.
- **`gc()` a cada época** — herança da época do CUDA. Em CPU, provavelmente só
  custa. Medir antes de remover.
- **Regressão apenas.** Nada no desenho impede classificação, mas a saída é
  `nn_linear(64, 1)` e as métricas são todas contínuas.

---

## Parte 3 — Prospecção

Cada item: **o que é**, **por que**, **como**, **custo** e **risco**. Ordenados
por quando a decisão precisa ser tomada, não por facilidade.

---

### A. Método — antes de publicar qualquer mapa

#### A0. O buffer está medindo o círculo errado

**Encontrado verificando, não revisando.** Reimplementei o relatório de
vazamento em Python, direto dos arquivos brutos, sem usar nada de `R/`, e
comparei com o que o pipeline reporta. As duas implementações concordam no que
o pipeline mostra — e discordam no que ele **não** mostra.

```
fold   treino    valid.    val no MESMO pixel      patch 15x15 sobreposto
1      20786     10393     0 (0,000%)              53  (0,51%)
2      20786     10393     0 (0,000%)              65  (0,63%)
3      20786     10393     0 (0,000%)              102 (0,98%)
```

**O 0% é verdadeiro.** Nenhum ponto de validação divide célula de raster com
treino — confirmado por implementação independente. Mas não era isso que o
buffer prometia.

**A geometria.** Dois patches de largura `w` dividem pelo menos um pixel
exatamente quando `|Δlinha| ≤ w−1` **E** `|Δcoluna| ≤ w−1`. Isso é um
**quadrado** (Chebyshev). `apply_buffer()` mede distância **euclidiana** — um
círculo. A fuga é a diagonal:

```
dois centros a 14 linhas E 14 colunas (w = 15)
  distância euclidiana = 14·√2 ≈ 19,8 células = 0,04447°
  buffer aplicado      =                        0,03369°   -> PASSA
  os patches           -> compartilham o pixel do canto
```

Minha regra — `buffer >= max(janela) × resolução` — garante o que parece
garantir **só se a distância for de Chebyshev**. Com euclidiana, o correto é:

```
buffer >= max(janela) × resolução × √2        (41% maior)
```

**Duas correções possíveis, e a segunda é melhor:**

1. Multiplicar o buffer por √2. Uma linha, funciona, descarta mais treino que
   o necessário (o excesso é toda a área entre o quadrado e o círculo
   circunscrito).
2. **Dar a `apply_buffer()` um argumento `metric = c("chebyshev", "euclidean")`
   com `"chebyshev"` como default.** É a geometria certa para patches
   quadrados, `buffer = w × res` volta a ser exato, e não desperdiça treino.
   `.near_any()` já divide o espaço numa grade — trocar o teste de
   `(dx² + dy²) ≤ b²` para `max(|dx|, |dy|) ≤ b` é uma linha, e os *buckets*
   continuam válidos.

**Gravidade: MÉDIA, não alta.** Contra 54,2% de sobreposição no split fixo,
0,5–1% é uma melhora de quase duas ordens de grandeza, e o run em andamento
continua sendo de longe a melhor validação que este projeto já produziu. Mas
é a diferença entre "praticamente separado" e "separado", e a segunda é a que
está escrita no código como garantia.

**Por que passou.** O `03` imprime só a linha `same raster cell` do relatório
de vazamento. O `fold_leakage_report()` **calcula** a sobreposição por janela —
o número estava lá, filtrado fora da tela. Correção de uma linha no `03`:
mostrar as duas.

**Não apliquei nenhuma das duas** — o run está em andamento e você pediu
prospecção, não execução. O que corrigi foi o **comentário em
`R/resample.R`**, que afirmava a regra errada. Comportamento inalterado.

#### A1. Teste espacialmente independente

**O quê.** O conjunto de teste deve ser separado pelo mesmo critério espacial
dos folds, não por amostragem aleatória estratificada.

**Por quê.** §2.1. Hoje o número que vai para o artigo é otimista e nada avisa.

**Como.** Duas opções, e elas não são equivalentes:

1. *Reservar blocos inteiros para teste no `01`.* Escolher, digamos, 15% dos
   blocos de 2° e marcar todos os seus pontos como `test`. Simples, e o teste
   passa a medir predição em terreno não visitado.
2. *Deixar o plano de reamostragem produzir o teste.* `spatial_folds()` já
   sabe repartir blocos; bastaria um `test_frac` que separa blocos antes de
   formar os folds. Mais elegante, e mantém a decisão num lugar só.

Prefiro a (2): o `01` não deveria decidir geografia — ele não sabe qual janela
você vai usar, e o buffer depende disso.

**Transição.** Os patches **não precisam ser reextraídos** — o split é índice,
não propriedade do dado armazenado. É exatamente o que o refactor comprou.
Muda só `dataset_role` no `patch_meta.csv`, ou nem isso, se o plano passar a
carregar o teste.

**Custo:** baixo (algumas horas). **Risco:** as métricas vão piorar, e isso é o
resultado correto. Vale antecipar o susto: a diferença entre o CCC de hoje e o
de depois é a medida de quanto do resultado era vizinhança.

#### A2. kNNDM em vez de blocos + buffer

**O quê.** Substituir (ou oferecer ao lado de) `spatial_folds()` o método
**kNNDM** — *k-fold Nearest Neighbour Distance Matching*, de Linnenbrink et al.
(2024), implementado no pacote **CAST**.

**Por quê.** Bloco + buffer é a geração anterior. O problema dele: o tamanho do
bloco e o do buffer são escolhas suas, e não há como saber se acertou. Eu
escolhi 2° medindo o balanceamento dos folds — que é um critério de
conveniência, não de validade.

kNNDM parte de outro lugar, e é o argumento que me convenceu: **a validação
certa depende de onde você vai prever.** Ele constrói os folds de modo que a
distribuição das distâncias ponto-de-teste → ponto-de-treino *durante a CV*
imite a distribuição das distâncias pixel-a-prever → ponto-de-treino *durante a
predição*. Minimiza a estatística W de Wasserstein entre as duas ECDFs.

Consequência elegante: se as amostras forem bem distribuídas, kNNDM converge
sozinho para um k-fold aleatório. Ele **não força** separação espacial quando
ela não é necessária — que é exatamente a crítica correta ao bloco cego.

**Como.** `CAST::knndm()` devolve índices de fold. Encaixa no desenho atual sem
tocar em mais nada:

```r
knndm_folds <- function(meta, k = 5, predpoints = NULL, ...) {
  res <- CAST::knndm(sf_points(meta), predpoints = predpoints, k = k, ...)
  # res$clusters -> vetor de fold por linha; embrulhar em .new_fold_plan()
}
```

Só mais um construtor ao lado de `holdout()` / `spatial_folds()` / `region_folds()`.

**Custo:** baixo (uma função + teste). **Risco:** dependência nova (`CAST`,
`sf`, `FNN`); e CRS projetado acelera muito (busca de vizinho via FNN) —
lon/lat exige matriz de distância esférica. Com 31 mil pontos globais isso
pode pesar; medir antes.

**Nota para o pacote:** manter `spatial_folds()` mesmo assim. Ele é
compreensível de cabeça, e num framework didático isso vale — mas o default
recomendado deveria passar a ser kNNDM.

#### A3. Separar early stopping de seleção

**O quê.** Reamostragem aninhada: o early stopping usa um conjunto interno; o
número reportado sai do conjunto externo, que o treino nunca viu.

**Por quê.** §2.4. Sem isso, `val_ccc` é um máximo sobre épocas, não uma
estimativa.

**Como (barato, e é o que eu faria).** Dentro de cada fold, separar ~15% do
treino como *conjunto de parada*. O early stopping monitora ele; a métrica
reportada continua vindo da validação do fold, que passa a ser genuinamente
não vista. Custa 15% dos dados de treino e nenhuma complexidade estrutural —
`build_fold_cache()` já aceita quantos papéis você quiser no índice.

**Como (caro e correto).** CV aninhada de verdade: laço externo escolhe, laço
interno tuna. Multiplica o custo por k. Para um grid honesto, é o padrão-ouro;
para este volume de CPU, é inviável hoje.

**Custo:** médio. **Risco:** treinar com 15% menos dados piora um pouco o
modelo. Recomendo medir o viés primeiro: comparar `val_ccc` na época escolhida
com `val_ccc` num conjunto de parada separado, num único fold. Se a diferença
for pequena, o problema é teórico e pode esperar.

#### A4. `one_se()` — a etapa 5 que já estava planejada

**O quê.** Em vez do melhor CCC médio, escolher o **modelo mais simples** cuja
média esteja a menos de um erro-padrão do melhor. É o `selectionFunction =
"oneSE"` do caret.

**Por quê.** O piso de ruído já está medido, e o primeiro run espacial mostrou
o problema com todas as letras: configs em 0,529 / 0,500 / 0,491 com sd entre
sementes de **0,0276**. A distância entre 1ª e 2ª é 0,029 — mal encosta no
ruído. Escolher a primeira é escolher a mais sortuda.

**Como.** Precisa de uma ordem de simplicidade, e ela é do domínio: eu
ordenaria por número de parâmetros (`test_architecture.R` já conta), o que
naturalmente prefere `gap` a `flatten`, janela menor e menos blocos.

```r
one_se <- function(by_config, metric = "val_ccc", complexity = "n_params",
                   maximise = TRUE) {
  best <- by_config[[paste0(metric, "_mean")]][1]
  se   <- by_config[[paste0(metric, "_se")]][1]
  ok   <- by_config[[paste0(metric, "_mean")]] >= best - se   # se maximise
  by_config[ok, ][which.min(by_config[[complexity]][ok]), ]
}
```

**Custo:** baixo. **Risco:** nenhum — é uma regra de seleção sobre uma tabela
que já existe. O trabalho de verdade foi medir o `se`, e isso está pronto.

---

### B. Desempenho

#### B1. Predição totalmente convolucional — exige uma decisão de arquitetura AGORA

Este é o item mais valioso e o mais delicado do documento. **Eu ia recomendá-lo
sem ressalva e a checagem me mostrou que metade estava errada.**

**A ideia.** Hoje a predição materializa um patch por pixel: com janela 15 e
181 canais, cada pixel vira um tensor de 181 × 15 × 15. Pixels vizinhos
compartilham quase todo o conteúdo — o fator de duplicação é **w² = 225**. É a
origem documentada dos picos de 30–37 GB de RSS no `05`, e da lentidão.

A alternativa clássica é rodar a pilha convolucional **sobre o tile inteiro,
uma vez**, e extrair a predição de cada pixel do mapa de saída. Cada posição é
convoluída uma vez em vez de w² vezes. As equivalências existem e são exatas:

| camada do modelo de patch | equivalente convolucional |
|---|---|
| `flatten` + `nn_linear(C·w·w, E)` | `nn_conv2d(C, E, kernel_size = w)` |
| `gap` + `nn_linear(C, E)` | `avg_pool2d(kernel = w, stride = 1)` + `conv 1×1` |
| `nn_linear(E, 128)` da cabeça | `nn_conv2d(E, 128, kernel_size = 1)` |
| `nn_batch_norm1d` em modo `eval` | `nn_batch_norm2d` (afim por canal, idêntico) |
| `nn_dropout` em modo `eval` | identidade |

**O problema que quase me escapou.** As convoluções são `padding = 1`. Num
patch de w×w, as posições da **borda** são calculadas com **zeros** fora do
patch. Numa passada convolucional sobre o tile, essas mesmas posições veriam os
**vizinhos reais**. E a borda importa: tanto `flatten` quanto `gap` consomem
todas as w² posições, não só o centro.

Ou seja: **a equivalência falha para o modelo como ele é hoje.** Uma FCN daria
resultado *diferente* — plausivelmente melhor, e irrelevantemente melhor,
porque não seria o modelo que foi treinado e validado.

**O que destrava.** Treinar com `padding = "valid"`. Aí não há zero nenhum, a
rede é genuinamente equivariante a translação, e a FCN é **exata**. Restrição:
cada conv de kernel 3 encolhe o mapa em 2, então `w ≥ 2L + 1` para L blocos —
com w = 15 e L = 3 sobram 9×9; com **w = 3 não cabe nem L = 2**, que é o mínimo
do grid atual.

Então a decisão real é de desenho:

| opção | predição | o que custa |
|---|---|---|
| manter `padding = "same"` | patch a patch, como hoje | lento, e a RAM é o gargalo |
| mudar para `"valid"` | ~w² mais rápida, exata | janela mínima maior; **retreinar tudo** |
| `"valid"` opcional no grid | os dois caminhos coexistem | complexidade; o `05` escolhe pelo que o modelo declara |

**Por que é urgente.** Só vale se decidido **antes** do treino final. Depois,
mudar o padding significa refazer o tuning inteiro.

**Argumento adicional a favor do `valid`, independente de velocidade:** o
zero-padding injeta uma borda artificial *idêntica em todas as amostras*. A
rede pode aprender a usá-la como referência — um artefato sistemático que não
existe no mundo. Não é hipotético; é o motivo pelo qual segmentação séria
evita padding ou compensa por ele.

**Recomendação.** Adicionar `conv_padding = c("same", "valid")` ao
`.cnn_param_space` e deixar o tuning comparar — o custo de descobrir é um grid,
e a informação decide os próximos anos do projeto. Se `valid` empatar em
acurácia, ele ganha por causa da predição.

**Custo:** médio para o parâmetro; alto para a FCN em si (reescrita do `05`).
**Risco:** a reescrita precisa de um teste de equivalência — predizer os mesmos
N pixels pelos dois caminhos e exigir diferença < 1e-5. Sem esse teste, não
vale começar.

#### B2. Onde o tempo realmente vai

Antes de qualquer paralelismo, medir. Do que já sabemos:

- `chunk_nrows` 200 → 1000 no `02` rendeu **4%** (5,24 h → 5,02 h). Minha
  hipótese (overhead por chamada do GDAL) estava errada: **domina o volume de
  I/O**.
- Por época, o laço faz `predict_loader` sobre a validação inteira e calcula
  todas as métricas — incluindo `DescTools::CCC` com intervalo de confiança,
  sobre 10.393 pontos, ~70 vezes por unidade. Candidato barato: CCC próprio e
  métricas completas a cada k épocas (a loss, que é o que o early stopping usa,
  continua toda época).
- Paralelismo sobre configs é tentador e provavelmente **errado** em CPU: o
  torch já usa 30 threads. Dois processos disputariam os mesmos núcleos. O
  paralelismo que compensa é o do `05` (processos independentes por tile), que
  já existe.

---

### C. O que falta para virar um pacote que outros usem

#### C1. Registro de modelos

**O item mais importante desta seção.** Hoje:

```r
build_cnn_from_config(cfg, n_channels)   # conhece uma arquitetura só
```

O que faz o caret ser caret não é o `trainControl` — é `method = "rf"` ao lado
de `method = "svmRadial"`, com a mesma chamada. Enquanto houver uma arquitetura
fixa, isto é "meu pipeline de CNN", não um framework.

**Como.** Um registro, no espírito do `parsnip`:

```r
register_model(
  name       = "dual_branch_cnn",
  build      = function(cfg, n_channels) { ... },   # devolve nn_module
  param_space = list(window_sizes = ..., embed_pool = ...),
  needs      = c("patches"),        # que tipo de entrada consome
  predict_fn = NULL                 # default: forward padrão
)
```

`make_tune_grid()` passa a sortear do `param_space` do modelo escolhido, em vez
de um `.cnn_param_space` global. Aí um usuário pode plugar uma U-Net, uma CNN
simples, ou um MLP sobre o pixel central — e comparar todos com o **mesmo**
plano de fold, as mesmas sementes e o mesmo piso de ruído. *Isso* é o produto.

**Custo:** médio-alto. **Risco:** é uma mudança de assinatura que toca
`tune_grid.R` e `train_cnn.R`. Fazer **antes** de empacotar; depois é quebra de
API.

#### C2. Objetos com classe, e verbos do R

Hoje tudo é tibble e lista nomeada. `fold_plan` já tem classe e `print` — e a
diferença de usabilidade é visível. Estender:

| objeto | métodos |
|---|---|
| `dlc_run` (retorno do resample) | `print`, `summary`, `plot`, `as_tibble` |
| `dlc_model` (unidade treinada) | `print`, `predict`, `coef`-like |
| `fold_plan` | ✅ já tem `print`; falta `plot` (mapa dos folds) |

`predict(fit, newdata = raster)` devolvendo um `SpatRaster` é a chamada que um
usuário de R espera. Hoje isso é um script de 897 linhas com variáveis de
ambiente.

#### C3. Estrutura de pacote

O trabalho mecânico, sem surpresas:

- `DESCRIPTION` / `NAMESPACE`; roxygen2 (os `#'` já estão lá — é conversão
  quase direta)
- `tests/testthat/` — os 9 arquivos atuais convertem quase 1:1; o `.report()`
  vira `expect_true()`
- Vinheta: o exemplo do `examples/` vira a vinheta, com um dataset mínimo
  embutido (essencial — hoje ninguém consegue rodar nada sem 16 GB de patches)
- `pkgdown` para o site; a `docs/design_decisions.md` já é conteúdo de vinheta
- **Decidir o que é público.** 64 funções públicas é muito. Meu corte: expor
  ~20 (construtores de fold, `run_*`, `summarise_*`, métricas, `predict`), o
  resto interno.

#### C4. Validação de entrada na fronteira

`check_point_contract()` é o começo certo — mas é a única. Um framework para
terceiros falha **na chamada**, com mensagem que diz o que fazer. Hoje um
`type_table` com colunas fora de ordem só quebra lá dentro, e um `tune_grid`
com coluna faltando quebra no `build_cnn_from_config`.

Recomendo `checkmate` na fronteira de cada função pública. Barato e muda a
experiência de quem usa.

#### C5. Regra geral que este projeto pagou três vezes

> **CSV é formato de apresentação.** Quando o código escreve para o próprio
> código ler, o formato tem que carregar o tipo.

Três incidentes, mesma causa: snapshot com separador decimal, `window_sizes`
`"3"` lido como número, `weight_decay` `"1e-04"` lido como texto. A correção
(RDS autoritativo + CSV legível) está aplicada no `comparison`. **Falta
aplicar o mesmo critério a todo o resto** — `patch_meta.csv`, `tune_grid.csv`,
`predictor_type_table.csv` são todos lidos pelo código.

Num pacote, isso vira decisão de arquitetura: um único `write_pair()` / `read_pair()`.

---

### D. Ciência — o que faria o pacote valer um artigo

#### D1. Área de Aplicabilidade sobre o *embedding* aprendido

**O quê.** A AOA (Meyer & Pebesma) delimita onde o modelo pode ser usado:
calcula um índice de dissimilaridade (DI) entre cada pixel a prever e os dados
de treino, no espaço dos preditores ponderado pela importância, e corta onde a
DI excede o que a CV observou.

**O problema de transferir direto.** A AOA assume preditores tabulares por
ponto. Aqui a entrada é um tensor 181 × 15 × 15 — distância euclidiana nesse
espaço não significa nada útil.

**A ideia.** Calcular a AOA no **espaço do embedding aprendido** — o vetor de
384 dimensões que `cnn_branch` produz. É o espaço em que o modelo de fato
opera, tem dimensão tratável, e a ponderação por importância fica implícita (a
rede já aprendeu o que importa).

Isso me parece publicável por si só: *"AOA para modelos de patch: dissimilaridade
no espaço de representação em vez do espaço de covariáveis"*. Ninguém que eu
tenha encontrado fez.

**Custo:** médio. **Risco:** o embedding muda a cada semente — o ensemble de
sementes precisaria de uma AOA por semente, ou de uma AOA sobre embeddings
concatenados. Tem substância de pesquisa aí.

#### D2. Incerteza calibrada

**O quê.** Hoje o `04` produz mediana e desvio de um ensemble de sementes.
`design_decisions.md` §11 já registra, honestamente, que esse desvio **não é um
intervalo de predição** — ele mede variação de inicialização, não o ruído
irredutível. Um mapa de "incerteza" que subestima é pior que nenhum.

**Como (ordem de esforço crescente):**

1. **Conformal split.** Separar um conjunto de calibração, calcular os resíduos
   absolutos, tomar o quantil (1−α) e somar/subtrair da predição. Dá cobertura
   garantida sem suposição de distribuição, e custa uma passada. **É o melhor
   retorno por esforço do documento inteiro.**
2. **Conformal quantile regression.** Treinar cabeças de quantil (pinball loss)
   e calibrar por conformal — intervalos que variam com o local, não largura
   constante.
3. **MC dropout / Monte Carlo conformal.** Publicado em 2025 para modelos
   espectrais de solo com deep learning.

**E validar a incerteza**, que quase ninguém faz: **PICP** (proporção de
observações dentro do intervalo nominal). Se você promete 90% e entrega 60%, o
mapa de incerteza está errado — e isso é mensurável.

#### D3. Importância de variáveis para patches (sua etapa 10)

Quando chegar lá, o desenho natural tem três níveis, e eles respondem perguntas
diferentes:

| nível | como | responde |
|---|---|---|
| **canal** | permutar o canal c em todos os patches, medir a queda | "este preditor importa?" |
| **espacial** | ocluir o anel a distância d do centro | "a vizinhança importa, ou só o pixel?" |
| **escala** | o `mean_gate` que `extract_gate_analysis()` já coleta | "qual escala domina, e onde?" |

O segundo é o que justifica a existência de uma CNN. Se ocluir tudo menos o
centro não piorar nada, a CNN não está usando contexto espacial — e um Random
Forest sobre valores pontuais faria o mesmo por 1% do custo. **Essa é a pergunta
que o projeto inteiro precisa responder, e ela ainda não foi feita.**

Eu faria esse teste **antes** da importância por canal. É barato (ocluir e
repredizer o conjunto de validação) e é falsificável.

#### D4. Pré-treino auto-supervisionado

**O quê.** Treinar o encoder sem rótulo (mascarar canais/regiões e reconstruir)
sobre patches de *qualquer* lugar do raster — há bilhões, contra 31 mil
rotulados — e depois afinar com os rótulos.

**Por quê.** A assimetria é brutal: 36.697 pontos rotulados (31.179 no pool de
treino) para 181 canais e 11 milhões de parâmetros. O early stopping dispara na época 7–14 em todas as
configs, o que é assinatura de sobreajuste rápido — o modelo esgota o que os
rótulos têm a dizer quase imediatamente.

**Risco:** é um projeto, não uma tarefa. Mas é o caminho com maior teto, e o
patch store já existe.

---

### E. Avaliado e NÃO recomendado — com o motivo

Registrado para não ser reproposto daqui a três meses.

| ideia | por que não |
|---|---|
| **Adotar `luz`** (API de alto nível do torch em R) | Traria early stopping, scheduler e callbacks prontos. Mas o laço atual tem comportamento específico — loss em espaço transformado a partir de uma passada só, métricas em unidade nativa por época, análise de gate — e está **provado por teste**. Trocar um motor testado por um genérico, para ganhar código que já funciona, é risco sem retorno. **Reavaliar se e quando o registro de modelos (§C1) existir** — aí `luz` reduz o custo de plugar arquitetura nova. |
| **GPU** | Já decidido e medido: não serve ao que você quer fazer. Não voltar. |
| **Integrar com `tidymodels`/`parsnip`** | O `parsnip` pressupõe entrada tabular (`data.frame`). Uma entrada N×C×H×W não cabe no contrato sem violência. Inspiração sim, integração não. |
| **`data.table` no lugar de `dplyr`** | O gargalo é I/O de raster e forward do modelo. Trocar o dialeto não move o ponteiro e reescreve código testado. |
| **Paralelizar configs no tuning** | O torch já usa 30 threads; dois processos disputam os mesmos núcleos. O paralelismo que compensa é o do `05`, por tile, e já existe. |
| **Comprimir o patch store** | Disco é irrelevante aqui (9 TB livres) e compressão custa CPU na leitura — que é o recurso escasso. Regra já estabelecida. |

---

## Parte 4 — Ordem que eu seguiria

Não é a ordem fácil; é a ordem em que uma decisão bloqueia a seguinte.

| # | item | por que agora |
|---|---|---|
| 1 | **`metric = "chebyshev"` no buffer** (§A0) | a garantia escrita no código não é a garantia entregue |
| 2 | **`conv_padding` no grid** (§B1) | **expira.** Depois do treino final, custa retreinar tudo |
| 3 | **Mostrar a sobreposição por janela no `03`** (§A0) | o número já é calculado e é filtrado fora da tela |
| 4 | **Teste para `spatial_overlap_report` e `calc_metrics`** (§2.2) | o pipeline afirma "0% de vazamento" com base numa função não verificada — e a verificação independente desta noite mostrou que a afirmação era verdadeira mas incompleta |
| 5 | **`evaluate_test = FALSE` no tuning** (§2.3) | uma linha; e cada run que passa consulta o teste mais 27 vezes |
| 6 | **`one_se()`** (§A4) | o piso de ruído já está medido e já mostrou que o ranking não separa |
| 7 | **Teste espacialmente independente** (§A1) | antes de qualquer número ir para o artigo |
| 8 | **README** (§2.5) | descreve um pipeline que não existe |
| 9 | **Oclusão espacial** (§D3) | barato, falsificável, e responde se a CNN se justifica |
| 10 | **Conformal + PICP** (§D2) | maior retorno por esforço em ciência |
| 11 | **kNNDM** (§A2) | melhora o que já funciona |
| 12 | **Registro de modelos** (§C1) | antes de empacotar; depois é quebra de API |
| 13 | **Empacotar** (§C3) | quando a API parar de mudar |

---

---

## Parte 5 — A lição de processo desta noite

O §A0 não apareceu lendo o código. Apareceu **reimplementando a pergunta em
outra linguagem, a partir dos arquivos brutos, e comparando**.

Revisar código é procurar o erro onde você já olhou. Reimplementar é perguntar
de novo por um caminho que não compartilha nenhuma suposição com o primeiro —
que é exatamente o que `check_patch_centres()` faz para a extração
(`terra::extract()` contra `cellFromXY`), e é por isso que aquela checagem é a
mais forte do arquivo.

**O padrão vale como regra:** toda afirmação que o pipeline faz sobre a própria
correção merece uma segunda implementação independente. Hoje existe uma
(centros de patch). As candidatas seguintes, em ordem de risco: o relatório de
vazamento (§2.2), `calc_metrics()` (§2.2), e a equivalência da FCN quando ela
existir (§B1) — que já nasce com esse teste como pré-requisito.

---

## Fontes consultadas

- Linnenbrink, Milà, Ludwig & Meyer (2024). *kNNDM: k-fold Nearest Neighbour
  Distance Matching Cross-Validation for map accuracy estimation.* Geosci.
  Model Dev. 17, 5897–5912 — <https://hannameyer.github.io/CAST/articles/cast03-CV.html>
- Meyer & Pebesma. *Predicting into unknown space? Estimating the area of
  applicability of spatial prediction models* — <https://arxiv.org/pdf/2005.07939>
  e vinheta AOA do CAST — <https://hannameyer.github.io/CAST/articles/cast02-AOA-tutorial.html>
- *Using Monte Carlo conformal prediction to evaluate the uncertainty of
  deep-learning soil spectral models.* SOIL 11, 553–563 (2025) —
  <https://soil.copernicus.org/articles/11/553/2025/>
- *geodl: An R package for geospatial deep learning semantic segmentation using
  torch and terra.* PLOS ONE (2024) —
  <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315127>
- `luz`: Higher Level API for torch — <https://mlverse.github.io/luz/>
