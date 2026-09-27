# Log de decisões

Registro do **porquê** de cada alteração, não só do quê. Uma mudança sem
motivo registrado é uma mudança que alguém reverte sem saber o que ela
protegia — e este projeto já perdeu semanas de CPU para um bug invisível.

Entradas em ordem cronológica inversa dentro de cada etapa. Decisões
**descartadas** também entram, com o motivo.

---

## Contexto: por que a refatoração começou

Auditoria completa do projeto em 2026-09-11 (duas páginas: diagnóstico do
exemplo SOC e roadmap do framework). Três achados definiram o trabalho:

1. **Vazamento espacial.** 33,2% dos perfis de teste dividem o mesmo pixel de
   250 m com um perfil de treino; 1 197 estão a menos de 1 m. As métricas
   publicáveis estão infladas por construção.
2. **Ranking dominado por ruído.** DP entre sementes = 0,0161 CCC; 1º e 2º
   lugares diferem por 0,001. O vencedor é, na prática, um sorteio.
3. **A raiz comum:** falta a abstração de reamostragem — a peça central do
   caret. Sem ela, CV espacial não é difícil, é inexprimível.

O plano tem 10 etapas. As etapas 1–3 são pré-requisito estrutural da 4
(resample), que é onde 1 e 2 se resolvem.

---

## Limpeza inicial — 2026-09-11

### Insumos movidos para `data/raw/`

| de | para |
|---|---|
| `outputs/soc_stock_layers_qc/` | `data/raw/soc_stock_layers_qc/` |
| `data/processed/.../full_data/*.gpkg` | `data/raw/` |

**Por quê:** `outputs/soc_stock_layers_qc/` (35 MB) mora em `outputs/` mas
**nenhum script deste repositório o gera** — é lido por `scripts/01` como
insumo do pipeline antigo de 20 km. Um `rm -rf outputs/` destruiria um GPKG de
origem impossível de regenerar daqui. Depois da mudança, `outputs/` significa
exatamente "descartável", que é a premissa de toda a limpeza.

Caminhos ajustados: `examples/01:44`, `scripts/01:25`.

### 27 GB de saídas apagados

**Por quê:** o usuário vai regerar tudo com a arquitetura nova, e manter
saídas da arquitetura antiga só criaria ambiguidade sobre qual artefato veio
de qual código. O que garante a volta ao estado anterior é a **capacidade de
regerar** (código + insumos + `set.seed`), não os arquivos — confirmado na
prática: o `01` reproduziu o split 25874/5540/5560 idêntico ao baseline.

### `.gitattributes`

**Por quê:** os arquivos novos saíram com LF e o repositório estava em CRLF,
gerando diffs sujos onde toda linha aparece alterada.

---

## Etapa 1 — rede de segurança (concluída, 60 asserções)

### `R/patches.R` — geometria unificada

**Por quê:** a álgebra de índice dos patches estava **duplicada**:
`.cell_index_mat()` no `02` e `build_patches_multi()` no `05`. Duas cópias da
mesma lógica podem divergir em silêncio, e o resultado seria alimentar a rede
com uma geometria e construir o mapa com outra — produzindo um mapa
plausível e errado, sem erro nenhum.

`02` e `05` passaram a consumir o módulo. **Isso era obrigatório antes do
teste**, senão o teste validaria uma cópia em vez do código real.

**Decisão de API revista durante a escrita:** a primeira versão expunha
validade e montagem como funções separadas, o que indexaria a strip duas
vezes. O `05` tinha uma otimização deliberada e documentada de indexar uma
vez só. A API final (`patch_band_assemble`, `patch_gather` + `patch_finish`)
preserva isso — uma indexação por janela, resultado reaproveitado.

### `tests/test_patch_geometry.R`

**Por quê:** nada verificava a geometria, e ela é a falha silenciosa mais
perigosa do projeto. Raster sintético com `valor = linha×1e6 + coluna×1e3 +
banda` torna o conteúdo correto conhecido analiticamente, pegando transposição,
troca linha/coluna, off-by-one e desalinhamento de canal numa tacada.

Também é a rede de proteção para a reescrita FCN da etapa 6, que pode quebrar
a geometria em silêncio.

### `tests/test_transform_loss.R`

**Por quê:** todo o early stopping do framework depende de
`transform_space_loss()` reproduzir a perda do torch — afirmação que o
docstring fazia e **nada verificava**. Se divergisse, todo modelo pararia na
época errada com curvas de aparência normal.

**Resultado:** diferença relativa 7,58e-08 (epsilon do float32). A afirmação
está validada e os runs anteriores pararam na época certa.

### `tests/helper.R` + localizador de raiz

**Por quê:** dois defeitos que só apareciam sob `source()`, presentes também
nos dois testes que já existiam:

1. `commandArgs("--file=")` é vazio sob `source()`, então `root` virava o
   diretório **pai** do projeto e os `source()` internos falhavam.
2. `quit(status = 1L)` **fecha a sessão do RStudio** numa falha.

`.report()` usa `stop()` quando `interactive()`, `quit()` só sob Rscript.

---

## Etapa 2 — correções sem mudança de número (concluída, 14 asserções)

### `clamp` no lugar de `pmax(., 0)` — `R/train_cnn.R:31`

**Por quê:** o piso de zero estava embutido em `predict_loader()`, *a* função
de inferência do framework. Correto para estoque de carbono; destrói metade
das predições, em silêncio e com métricas plausíveis, para pH, log-ratio,
temperatura ou qualquer alvo centrado. Default `c(0, Inf)` preserva o
comportamento anterior exatamente (há teste que compara os dois).

### Validação de `gate_type` e `embed_pool` na construção

**Por quê:** `gate_type` desconhecido caía no `NULL` do `switch`, pulava a
construção do gate e só explodia no `forward` — depois do modelo construído e,
num grid, minutos de treino depois. `embed_pool` era pior: qualquer valor
diferente de `"gap"` virava `flatten` em silêncio pelo `identical()`, então um
typo mudava a arquitetura sem avisar.

Janela única continua ignorando `gate_type` — configs antigas carregam
qualquer placeholder ali e precisam continuar funcionando.

### `status = "failed"` sendo escrito

**Por quê:** config que falhava fazia `next` sem escrever linha, ficando
indistinguível de uma que nunca rodou. E o `resume` filtra por
`status == "success"`, o que implicava que um `"failed"` deveria existir — a
coluna só podia valer um valor. Agora a linha entra com a mensagem de erro, e
o ranking só numera os que tiveram sucesso.

**Bônus:** a linha antiga de um config é removida antes da nova, então retomar
um run não duplica linhas.

### `model` não é mais devolvido por `train_one_cnn()`

**Por quê:** nada consumia esse campo (verificado por busca), e como `result`
só é sobrescrito na iteração seguinte, o modelo da config anterior ficava vivo
durante o treino da próxima.

### `build_cnn_from_config()` usa `names()` + `[[ ]]`

**Por quê:** `cfg$embed_pool` num tibble sem a coluna devolve `NULL` **e** emite
`Unknown or uninitialised column` — ruído num caminho que é deliberadamente
opcional (retrocompatibilidade com grids antigos).

---

## Etapa 3a — QC separado do escalonamento (concluída, 12 asserções)

### `R/preprocess.R`

**Por quê:** `.scale_band_vec()` fundia duas coisas que precisam viver em
pontos diferentes do pipeline:

| metade | depende do fold? | quando roda |
|---|---|---|
| QC (piso físico → NA, clamp de intervalo) | **não** | extração, uma vez |
| escalonamento (μ/σ estimados) | **sim** | montagem do tensor |

Um valor fisicamente impossível é impossível independente de quem está no
treino; μ e σ não são. Enquanto estavam fundidos, CV k-fold exigiria k
re-extrações dos rasters — horas cada. Separados, custa uma broadcast por fold.

**Simplificação que apareceu:** todo tipo de preditor vira a mesma operação
afim, só mudam as constantes — contínuo `(x−μ)/σ`, percentual `(x−0)/100`,
dummy `(x−0)/1`. Então `scale_patches()` é uma operação para os 187 canais,
não um `switch` por método.

**Generalização do QC:** as regras eram literais dentro de duas funções.
`make_qc_table()` produz regras por preditor (`na_below`, `clamp_lower/upper`),
com defaults idênticos ao comportamento anterior.

### `tests/test_preprocess.R`

**Por quê:** a equivalência `scale-then-extract == extract-then-scale` é a
premissa de toda a etapa 3. Provada em dado sintético — necessário porque os
17 GB de patches reais foram apagados na limpeza, então não há "antes" contra
o que comparar. Virou teste de verdade em vez de comparação contra artefato.

Diferença esperada e documentada: o caminho novo escalona **em float32** (no
tensor), o antigo escalonava em double e convertia depois. Diferença relativa
~1e-7 — é onde a conversão acontece, não regressão.

---

## Etapa 3b — reescrita do 01 e do 02 (em andamento)

### `01`: removidos os 6 CSVs por split

**Por quê:** verificado quem consumia cada um antes de decidir.

- **`*_scaled.csv` (3 arquivos, ~127 MB): nenhum código lia.** Nem o `02` — o
  comentário dele dizia explicitamente que lia o `_raw` porque as colunas
  pré-escalonadas "seriam ignoradas". Já eram peso morto.
- **`*_raw.csv`:** só o `02` (reescrito nesta etapa) e 2 checagens do `99`.
  São um `filter()` do `full_modeling_dataset_raw.csv`.
- **`06_avaliacao_grafica.R`:** não toca em nada do `01`. Zero impacto.

O ganho decisivo não é disco: com escalonamento por fold, **"o conjunto
escalonado" deixa de existir como objeto único**, e manter `train_scaled.csv`
seria manter um arquivo afirmando algo que deixou de ser verdade. Mantê-los
carregaria para dentro da etapa 4 exatamente a estrutura que a etapa 4 existe
para derrubar.

### `01`: relatório de risco de canal (`channel_risk.csv`)

**Por quê:** três famílias de canal já quebraram o mapa deste projeto e
**nenhuma aparece como erro ou métrica ruim**:

- `constant` — zero informação nos pontos, mas não constante no mundo
- `near_constant` — mesmo risco, menor
- `has_na` — NA nos pontos; a família que esvaziou o mapa a 250 m

O filtro de sd degenerado que já existia **não pode** pegar os constantes,
porque dummy recebe sd = 1 e percentual sd = 100 por definição, nunca um sd
estimado.

### `01`: `qc_table.csv` exportado

**Por quê:** as regras de QC eram literais duplicados dentro do `01` e do `02`,
mantidos em sincronia na mão — e **foi essa duplicação que deixou o bug do
percentual passar**. Agora o `01` decide e o `02` obedece.

### `01`: drop dos 6 canais constantes

Removidos: `geology_ice_and_glaciers`, `soil_class_fao_islands`,
`soil_class_fao_no_data`, `terrestrial_habitat_deep_ocean_floor`,
`pnv_moss_and_lichen`, `pnv_open_forest_deciduous_needleleaf`.
187 → 181 canais.

**Por quê:** são 0 em todos os 36 974 perfis, mas valem 1 sobre geleira, ilha,
área sem dado de solo e oceano profundo. Como a entrada é sempre 0 no treino,
o gradiente dos pesos é sempre 0 — e com `weight_decay = 0` (o do cfg_014)
esses pesos **permanecem na inicialização aleatória para sempre**. Cada canal
aplica então um viés arbitrário e *diferente por semente* exatamente sobre os
pixels que estão fora de tudo que a rede viu.

Hipótese não comprovada, mas consistente: pode contribuir para a dispersão
entre sementes de 0,52 medida nas tiles árticas.

**Alternativas descartadas:**
- *Manter os 187* — o custo direto é trivial (3 456 pesos mortos), mas o viés
  aleatório nas regiões extrapoladas não é.
- *Dropar automaticamente qualquer canal constante* — perigoso num framework:
  "constante nos pontos" ≠ "constante no mundo". A detecção avisa; o drop é
  explícito e fica registrado no script.

### `01`: `force_as_percentage` esvaziado

**Por quê:** continha os dois PNV que agora saem como constantes. O override
**nunca teve efeito prático** para eles: um canal que é 0 em todos os perfis
não tem tipo para errar. Gancho mantido, porque a auto-detecção pode
genuinamente ler mal uma classe percentual rara assim que perfis a ativarem.

### `02`: reescrito

**Uma passada em vez de três.** Os splits são subconjuntos disjuntos de pontos
sobre o **mesmo** raster; ler as 187 bandas três vezes era repetição pura.
Corta a I/O em 3×.

**Patches crus.** Consequência direta da etapa 3a — ver acima.

**Matriz de culpa por canal (`channel_invalidation.csv`).** A salvaguarda
central. `valid_common` é um AND sobre 187 canais, então sozinho nunca pode
dizer **qual** derrubou um ponto. Agora reporta `n_invalidated` e
`n_sole_cause` (quantos pontos você recupera removendo só aquele canal), com
aviso explícito se o pior passar de 1%.

**Por quê importa tanto:** a regra de janela completa transforma um pixel NA
num buraco de até 15×15 ao redor. A cobertura de um tile da Amazônia foi de
**0,72% → 100%** depois que o clamp de percentuais foi corrigido — e só
apareceu depois de rodar o pipeline inteiro a 250 m. Este relatório teria
pegado isso no `02`, em minutos.

**Um arquivo por janela, float32.** ~8,6 GB em vez de 17, e o `03` carrega só
as janelas que o grid usa. Cada tensor é relido e conferido depois de salvo,
porque `torch_save` de multi-GB é caminho novo aqui.

### `05a_test.R:204`: `|>` → `unlist()`

**Por quê:** regra do projeto é `%>%`, nunca o pipe nativo. Sobrou uma
ocorrência em `_scratch_1km_test/`, que é código duplicado e sai na etapa 7.

---

### `R/dataset.R` — o patch store e o cache do fold

**Por quê:** é a peça que faltava entre o `02` (que escreve patches crus) e o
laço de treino (que precisa de tensores escalonados e divididos). Três
conceitos que antes estavam implícitos viraram explícitos:

| função | o que torna explícito |
|---|---|
| `load_patch_store()` | carrega **só as janelas pedidas** — a razão de ser do formato um-arquivo-por-janela |
| `split_index_from_meta()` | o split é um **índice**, não propriedade do dado |
| `build_fold_cache()` | o escalonamento é do **fold**, estimado de `index$train` |

`split_index_from_meta()` é o caso degenerado (holdout fixo, lido de
`dataset_role`) e é o que o pipeline usa hoje. Um construtor de folds espaciais
encaixa exatamente no mesmo slot, sem mudar mais nada — é assim que a etapa 4
entra.

**Validações que ele impõe em vez de assumir:** ordem de canais entre
`predictor_type_table.csv` e o manifest; shape de cada tensor carregado;
índice disjunto (ponto em dois splits é vazamento) e completo (ponto em nenhum
é dado descartado em silêncio); escalonamento degenerado no fold. E recusa um
store que tenha sido gravado **com** escalonamento aplicado — que seria um
store amarrado a um único split.

### `scale_patches(inplace = )`

**Por quê:** o tensor 15×15 tem ~6 GB em float32. `(x - c) / s` aloca uma
cópia; `x$sub_(c)$div_(s)` não. Em `build_fold_cache` o clone é deliberado (o
store precisa sobreviver cru para o próximo fold) e o escalonamento é in-place
sobre o clone — pico de uma cópia em vez de duas.

### `run_cnn_tuning()` recebe `cache` em vez de `patches`

**Por quê:** o cache era construído **dentro** do laço de grid, o que obrigava
o escalonamento a ser propriedade do dado armazenado. Passando o cache pronto,
o escalonamento passa a pertencer ao fold e o laço só consome. Quando a etapa 4
chegar, este argumento vira "o cache do fold atual" e o laço não muda uma linha.

`.build_tensor_cache()` e `.make_loaders()` foram removidos (obsoletos).
`.make_loaders_from_cache()` passou a usar `patch_window_key()` — a mesma regra
de nome que o store usa nos arquivos, para que uma janela não possa ser
procurada sob um nome que ninguém escreveu.

### `03`: grid antes do carregamento

**Por quê:** inversão de ordem exigida pelo formato novo. `windows_needed` vem
do grid, e o store carrega só essas janelas — um grid que nunca usa 15×15 não
paga ~6 GB de RAM por ela.

**Mudança de comportamento registrada:** o escalonamento agora é estimado dos
pontos que **efetivamente treinam** (pós-QC de janela), enquanto o
`predictor_scaling.csv` do `01` usava todos os pontos de treino, incluindo os
~0,8% depois descartados pela regra de janela completa. A diferença é pequena,
mas esta é a versão honesta — e a única que generaliza para um fold. O
escalonamento efetivamente usado é gravado em `scaling_holdout_train.csv`.

### `99_check_pipeline.R`: checagens migradas e duas novas

**Por quê:** as checagens antigas apontavam para arquivos que deixaram de
existir. Aproveitei para automatizar as duas proteções novas:

- **`nenhum canal constante sobreviveu ao drop`** — FAIL se um canal constante
  passar. Antes era só uma mensagem que alguém precisava ler.
- **`pior canal, % de pontos invalidados`** — WARN acima de 1%, FAIL acima de
  5%, nomeando o canal. Complementa a checagem que já existia (`pct_removed`,
  que o comentário do próprio script chama de "A CHECAGEM MAIS IMPORTANTE DESTE
  SCRIPT INTEIRO"): aquela diz **quanto** se perdeu, esta diz **qual canal**
  perdeu.
- **`patches gravados SEM escalonamento`** — FAIL se o store vier pré-escalonado.
- `n_channels` deixou de ser comparado com o literal `187` e passou a ser
  comparado com o que o `01` preparou. Dropar um preditor problemático é ação
  legítima e não deve quebrar um check.

**Guarda de escopo:** `ptype` é lido dentro do bloco `if (all_01_exist)`, e a
etapa 02 o usava. Se a etapa 01 estivesse incompleta, a 02 quebraria em vez de
reportar. Agora degrada para WARN.

### Ferramenta de verificação (scratchpad, fora do repo)

**Por quê registrar:** meu verificador de balanceamento acusou
`R/tune_grid.R` como quebrado. Era **falso positivo** — `` `[[` `` é um nome de
função entre crases, não indexação. O arquivo está intocado. Registrado porque
a conclusão errada aqui seria "mexeram no tune_grid".

---

### Desacoplamento do framework em relacao ao exemplo — 2026-09-12

**Por quê:** lembrete explicito do usuario de que `R/` e para **terceiros**
usarem, com outro dado e outro dominio. Ele sugeriu que isso poderia esperar a
etapa 9 (virar pacote); discordei e registro o motivo: a etapa 9 resolve
`DESCRIPTION`/`NAMESPACE`/roxygen, que e mecanico. O que ela **nao** resolve e
assinatura, default, e para onde a mensagem de erro aponta — decisoes tomadas
na hora de escrever a funcao. Refaze-las depois e reescrever.

Levantamento encontrou tres acoplamentos, **dois deles introduzidos por mim**
no dia anterior:

| acoplamento | onde | correcao |
|---|---|---|
| `na_below` default era a regra de temperatura DESTE dataset | `preprocess.R:47` | default `NULL`; o exemplo passa a regra |
| erro mandava rodar `02_extract_patches.R`, arquivo que so existe no exemplo | `dataset.R:47,56` | aponta para a FUNCAO e para o formato esperado |
| nomes de coluna cravados (`profile_id`, `sample_id`, `target_native`, `target_transform`) | `train_cnn.R`, `dataset.R` | virou contrato explicito e validado |

**Alternativa descartada para o terceiro:** um argumento por nome de coluna
(`id_col=`, `target_col=`, ...). Espalharia a mesma complexidade por toda
assinatura do pacote. Um contrato documentado e checado na entrada concentra
isso num lugar so. O que nao pode acontecer e o usuario **descobrir** o
contrato por um erro obscuro dentro do laco de treino — dai
`check_point_contract()`, que lista o que falta e o que cada coluna significa.

Comentarios de `R/` tambem deixaram de citar "script 01/02/05". Hoje nenhum
arquivo em `R/` referencia o exemplo.

**Pendente relacionado:** um teste que rode um ciclo completo usando SOMENTE
`R/*.R` + dado sintetico, sem tocar em `examples/`. Se nao rodar, o framework
nao e usavel por terceiros — e isso vira verificavel em vez de opiniao.

---

### Incidente: 5 h de extracao perdidas por uma verificacao frageil — 2026-09-12

**O que aconteceu:** o `02` extraiu tudo (5,24 h), gravou `patches_w15.pt`
inteiro e **abortou** na verificacao. O arquivo estava correto:
5.977.941.502 bytes contra 5.977.941.300 de dados + 202 de cabecalho, shape
`36697 x 181 x 15 x 15`, **zero celulas nao-finitas**. Falso positivo.

**A verificacao que eu escrevi:**

```r
ok <- identical(as.integer(back$shape), c(n_valid, n_channels, w, w)) &&
      as.logical((back[1,1,1,1] == x[1,1,1,1])$item()) && ...
if (!isTRUE(ok)) stop("Verification failed for ", f)
```

Quatro defeitos, em ordem crescente de gravidade:

1. `identical()` entre tipos — um `double` de um lado reprova numeros iguais.
2. `==` com NaN — `NaN == NaN` e `FALSE`, entao uma celula NaN amostrada
   reprova um arquivo perfeito.
3. tres condicoes em `&&` — na falha, nao da pra saber qual.
4. **`stop()` numa checagem de sanidade depois de 5 h de trabalho.** Esse e o
   erro de verdade. Uma checagem que pode dar falso positivo nunca deve
   abortar trabalho caro.

**Agravante, e culpa minha:** o script de recuperacao que escrevi carregou o
tensor de 6 GB para diagnosticar, numa sessao que ja segurava ~5 GB de arrays
e 5 h de fragmentacao de heap. A sessao abortou (`R Session Aborted`) e os
arrays de 9x9 e 3x3 se perderam. Memoria de tensor torch vive **fora** do heap
do R, entao `rm()` + `gc()` nao a devolve na hora. O diagnostico devia ter
vindo DEPOIS de gravar o que estava em risco, ou nem ter carregado o tensor.

**Correcoes no `02`:**

| antes | depois |
|---|---|
| verifica carregando o tensor e comparando celulas | verifica pelo **tamanho do arquivo**, derivado do shape — nao carrega nada |
| `stop()` na falha | reporta e segue; o que foi extraido ja esta no disco |
| metadados gravados por ultimo | `patch_meta.csv` gravado **primeiro** — define quais pontos sobreviveram e ancora os tensores |
| sem retomada | janela ja no disco com tamanho correto e **pulada** |
| `patch_list[[key]][valid_idx, , , ]` aloca um segundo array double inteiro (12 GB) | converte para tensor **antes** de fatiar — pico ~6,4 GB menor (29,3 → 22,9 GB) |

**Por que a retomada e segura:** o tamanho esperado usa o `n_valid` calculado
no run atual. Se o conjunto de pontos validos mudar, o tamanho nao bate e o
arquivo e regravado. Um store com w15 de um conjunto e w09 de outro nao passa
despercebido.

**Otimizacao descartada:** inferir `n_valid` do tamanho do arquivo existente e
pular a **alocacao** do array w15 (nao so a gravacao), reduzindo o pico de
16,9 para 4,9 GB. Descartada porque a validade de um ponto depende da janela
15: sem extrai-la, o conjunto valido seria outro. Inferir o conjunto do
tamanho do arquivo suporia que codigo e dados nao mudaram — exatamente a
classe de suposicao silenciosa que ja custou caro aqui.

---

### Contrato movido para `R/utils.R` — 2026-09-12

**Por quê:** `check_point_contract()` nasceu em `R/dataset.R` e era chamada por
`predict_loader()` (em `R/train_cnn.R`). Sob `source()` a ordem importa, e o
`test_validation.R` — que nao carrega `dataset.R` — quebrou com "nao foi
possivel encontrar a funcao". Num pacote isso nao existiria (mesmo namespace),
mas uma funcao que **dois** modulos precisam nao pode morar dentro de um deles.

Mudou para `R/utils.R`, a fundacao que todo ponto de entrada carrega primeiro.
O motivo esta escrito no proprio arquivo, para ninguem "organizar" de volta.

### `test_preprocess.R` dependia do default acoplado

**Por quê registrar:** essa falha foi a evidencia de que o desacoplamento valia
a pena. O teste chamava `make_qc_table(predictors, clamp_range = "pnv_shrubs")`
sem `na_below`, contando com o default `surface_temperature_celsius$ = -100`
que tinha acabado de ser removido por ser regra do dataset de SOC, nao do
framework. O teste caiu na hora, provando que o acoplamento era real e estava
em uso.

Corrigido passando a regra explicitamente (como o `01` faz), e aproveitado para
cobrir o outro lado: `no_rules_means_no_qc` assere que, sem regra nenhuma, o
framework nao inventa QC. 12 -> 13 assercoes.

**Licao de processo:** as duas regressoes deste dia foram pegas pelo `run_all`
em 3,7 segundos, antes das 5 h do `02`. O custo de rodar a suite antes de
qualquer script caro e proximo de zero; o custo de nao rodar ja foi medido.

---

## 2026-09-13 — armazenamento dos patches: o incidente do `torch_save`

### O que aconteceu

Troquei o armazenamento dos patches de `saveRDS` (arrays double, ~17 GB) por
`torch_save` (tensores float32, ~8,6 GB) para economizar disco. O R passou a
morrer **depois** de terminar a escrita, e uma vez gravou um arquivo de 5,98 GB
cujos ultimos 4,3 GB eram zeros — sem erro, sem aviso.

Passei varias rodadas atras de RAM. Estava errado, e voce apontou: *"acho q o
problema n é a ram nao tem algum problema no seu codigo"*.

### A causa, medida

O serializador do `torch_save` usa offset **inteiro de 32 bits**. Medido no
limite, com o script guardado em `tests/_diag_torch_save_limit.R`:

| bytes | resultado |
|---|---|
| 2.147.479.648 | grava e rele correto |
| 2.147.487.648 | mata a sessao |

`2^31 = 2.147.483.648` cai exatamente entre os dois. Nao e pressao de memoria:
e overflow silencioso de offset. Acima do limite o arquivo sai com tamanho
plausivel e conteudo truncado — o pior modo de falha possivel.

### Decisao

Voltar para `saveRDS`, e **nao** tentar contornar (chunking, split por canal,
formato proprio). Motivos, nesta ordem:

1. **O ganho nao existia.** *"ese tipo de problema de armazenaemnto é
   irrelevante!"* — 8,6 GB economizados num disco com 9 TB livres, ao custo de
   um formato que corrompe em silencio. Trocar corretude por disco e um mau
   negocio em qualquer proporcao.
2. **Framework para terceiros.** Quem usar isto com outro dado nao vai saber do
   limite de 2^31. Um default que corrompe acima de um tamanho nao documentado
   nao pode ser o default.
3. **`saveRDS` e o mecanismo mais simples comprovado.** Sem dependencia de
   torch para ler o store, sem limite de 2 GB, e o array volta exatamente como
   entrou.

O `torch_save` continua em uso para o que ele serve — **pesos de modelo**, que
sao ordens de grandeza menores. Por isso `safe_torch_save()` ganhou uma guarda
que **recusa** a escrita acima de 2^31 bytes em vez de deixar corromper: se
algum dia um modelo chegar la, o usuario ve um erro, nao um arquivo com zeros.

### O que a verificacao aprendeu

A verificacao anterior abortou 5 h de extracao num falso positivo: um
`if (!isTRUE(ok)) stop(...)` sobre um `ok` que combinava `identical()` de
tipos, `==` que podia dar NaN, e tres `&&`. Um dos termos deu NA e tudo foi pro
ralo.

Tres mudancas, todas no `02`:

- **metadados gravados PRIMEIRO**, antes dos patches — se a extracao cair, o
  que ja existe continua legivel;
- **retomavel**: janela ja em disco com o tamanho esperado e pulada, entao uma
  queda custa a janela corrente, nao as 5 h;
- **verificacao reporta, nunca aborta** — a decisao de descartar 5 h de
  trabalho e do usuario, nao de um `&&`.

E a regra virou teste (`test_patch_store_io.R`, 15 assercoes): **nunca conferir
uma escrita so pelo tamanho do arquivo — reler e comparar o CONTEUDO**. Foi
exatamente o tamanho plausivel que escondeu os 4,3 GB de zeros.

### `chunk_nrows` 200 -> 1000: 4%, nao o que eu previ

Aumentei o chunk apostando que o custo por chamada do GDAL dominava. Medido:
5,24 h -> 5,02 h. **4%.** A hipotese estava errada — o que domina e o volume de
I/O, nao a frequencia das chamadas. Fica registrado porque o valor 1000 no
codigo parece deliberado e merece dizer que rendeu pouco: quem for otimizar o
`02` de verdade tem que atacar leitura de raster, nao tamanho de bloco.

---

## 2026-09-13 — `R/diagnostics.R`: checagens sobre o RUN, nao sobre o codigo

Os testes em `tests/` provam que o **codigo** esta certo, em dados sinteticos,
em segundos. Nao provam que **este run** esta certo nos dados reais. Sao
perguntas diferentes, ambas baratas, e nenhuma substitui a outra.

**`check_patch_centres()`** — a mais forte, e a que nada fazia. O centro de
cada patch tem que ser igual ao valor da tabela de pontos: e a mesma celula do
mesmo raster por dois caminhos independentes (`terra::extract()` no 01;
`cellFromXY` -> `patch_cell_index()` no 02). Erro de CRS, troca linha/coluna,
off-by-one, reordenacao de canal ou diretorio desatualizado quebram a
igualdade. Le so a menor janela (o centro e o mesmo em todas). No run limpo:
**6.642.157 celulas conferidas, 0 divergentes**.

**`spatial_overlap_report()`** — deixa a metrica de vazamento visivel a cada
run, em vez de ser um achado de auditoria que se perde. WARN, nunca FAIL:
split aleatorio e escolha legitima; o inaceitavel e nao saber. Custa uma
passada e nenhum pacote novo (hash de row/col, O(n)).

**`write_run_snapshot()` / `compare_run_snapshot()`** — "mudou alguma coisa
desde a ultima vez?" respondido por diff, nao rolando a tela. "Tudo identico" e
o resultado que mais se quer e o mais dificil de confirmar no olho.

### Dois WARN do proprio 99 que eram bugs meus

O run limpo fechou em PASS 41 / WARN 4 / FAIL 0 — e **dois dos quatro WARN nao
eram achados, eram defeitos do verificador**:

1. **Coluna renomeada.** O `patch_files.csv` traz `status`
   (`written`/`kept`); o 99 continuou lendo `verified`, nome da versao que
   gravava tensores. Resultado: "0/3 verificados" num store perfeito, mais dois
   *Unknown or uninitialised column*. Um check que le a coluna errada e pior
   que check nenhum — gasta atencao num alarme falso. Agora le `status` e diz
   quais status apareceram.

2. **Locale comendo o decimal.** `as.character(31.190645)` grava com **ponto**;
   `read_csv2()` com locale `;`/`,` le o ponto como separador de **milhar** e
   devolve `31190645`. Quatro valores apareciam como "changed" sem nada ter
   mudado — exatamente o ruido que o snapshot existe para eliminar. Corrigido
   em `.read_snapshot()`, que le **tudo como texto**: snapshot e comparado como
   texto, entao a leitura tem que devolver texto. O snapshot gravado com o bug
   foi apagado para nao envenenar o proximo diff.

Restam **2 WARN, ambos legitimos**: validacao 27,06% e teste 28,08%
compartilhando pixel com o treino. Sao o motivo de a opcao de split espacial
existir (etapa 4).

---

## 2026-09-13 — tres defeitos que so aparecem rodando

O run limpo fechou PASS 42 / WARN 3 / FAIL 0 e expos coisas que nenhum teste
sintetico pegaria, porque sao sobre a *saida* do verificador:

### 1. O 99 avisava por ter sido consertado (alca fechada)

O snapshot incluia `99_n_pass`, `99_n_warn` e `99_n_fail` — os contadores do
proprio 99 — e os comparava com o run anterior. Consertar dois checks quebrados
mudou 41→42 e 3→2, e o diff acusou "2 de 22 valores mudaram", gerando um WARN.
**Melhorar o pipeline produzia um aviso.**

Os contadores sao *derivados* de todas as outras chaves: se uma metrica real
mudar, ela ja aparece no diff sozinha. Continuam **gravados** (sao o resumo do
run, valem para o historico) mas saem do diff via `exclude=`. Um alarme que
dispara quando voce faz a coisa certa treina o usuario a ignorar alarmes.

### 2. 22 linhas de ruido de locale no meio do relatorio

`read_csv2()` emite *"Using ',' as decimal and '.' as grouping mark"* a cada
chamada — sobre o locale que **nos** escolhemos, entao sem informacao nenhuma.
Com 22 leituras, o relatorio saia picotado.

Silenciado uma vez em `safe_read_csv2()` (`R/utils.R`), par de leitura do
`safe_write_csv2()` que ja existia. Fica no framework, nao no exemplo: quem usar
isto com outro dado tem o mesmo locale e o mesmo ruido.

### 3. `message()` e `print()` colando na mesma linha

`message()` escreve em **stderr**, `print()` em **stdout**; no console do
RStudio os dois se juntam — dai a saida `...mudaram:# A tibble: 2 × 4`.
`print_snapshot_diff()` passou a usar `cat()`: relatorio tem que sair todo pelo
mesmo canal para a ordem ser garantida.

**O padrao dos tres:** nenhum e erro de calculo — todos sao o verificador
comunicando mal. Num framework, a saida da checagem *e* a interface; ruido nela
custa a confianca em tudo que ela afirma.

---

## 2026-09-13 — etapa 4: reamostragem, sementes e o piso de ruido

### `R/resample.R`: o plano e um objeto, nao uma lista de indices

`holdout()`, `random_folds()`, `spatial_folds()`, `region_folds()` devolvem um
`fold_plan` que carrega o metodo, os parametros e a atribuicao por linha. Uma
lista crua de indices nao sabe dizer como foi construida, e um run cujos folds
nao podem ser descritos e um run que nao pode ser defendido. O plano e gravado
**antes** do treino, junto dos resultados.

**Default = `holdout()`**, deliberadamente: reproduz exatamente o que o pipeline
fazia antes de existir reamostragem. Ligar CV e escolha do usuario, nunca algo
que uma versao nova fez com ele.

**O teste nao e reamostrado.** As linhas `test` continuam `test` em todo fold.
Teste que muda de fold ja foi visto por algum modelo do ensemble — ai nao e
teste, e uma terceira validacao com nome enganoso.

### O achado: blocar sozinho NAO separa nada

O teste sintetico (12 sitios a 50 km, 30 pontos cada) mostrou que
`spatial_folds()` sem buffer deixava **33% dos pontos de validacao com um
vizinho de treino a menos de 1 km**. Causa medida: a grade de blocos e
desenhada **no mapa**, numa origem que nao tem relacao com onde os pontos
estao, entao uma borda cai dentro de um cluster — **4 dos 12 sitios cortados**.
Metade do sitio treina, metade valida, a 400 m de distancia.

Eu tinha escrito a assercao como `< 0.01` e ela falhou. **A assercao e que
estava errada, nao o resultado**: afirmar que blocar basta seria o teste
mentindo para proteger o metodo. Virou `blocking_alone_still_leaks`.

O que faltava era o **buffer** (`apply_buffer()`), e a regra que o amarra ao
que ja mediamos no 99:

> Dois patches de largura `w` a resolucao `res` dividem pelo menos um pixel
> quando os centros estao a menos de `w * res`. Entao
> **`buffer >= max(janela) * resolucao`** garante que nenhum patch de treino
> divide um unico pixel com um patch de validacao — exatamente a quantidade que
> o 99 reporta como "patches overlap (WxW)". Para 250 m com janela 15: **3750 m**.

E o buffer **custa dado de treino**; o plano guarda e imprime quanto (11,1% por
fold no fixture). Descartar treino para comprar validacao honesta pode ser o
trade certo ou nao, mas tem que ser visivel.

### O resultado que vale o arquivo inteiro

Bufferizar um k-fold **aleatorio** sobre dados agrupados **remove todos os
pontos de treino**. Todo cluster tem ponto de treino e de validacao, entao todo
ponto de treino tem um vizinho de validacao, entao o buffer leva todos. Isso e
o que sobra da CV aleatoria em dados agrupados depois que se tira o vazamento:
nada.

**Blocar e bufferizar nao sao alternativas** — blocar e o que torna o buffer
pagavel, colocando clusters inteiros de um lado so da linha. E isso tem que
falhar **alto**: treino vazio descoberto cinco horas adentro de um run e da
mesma familia dos erros que este projeto existe para impedir.

### A unidade de trabalho virou `(config, fold, semente)`

Cada uma tem `unit_id`, checkpoint e historico proprios. Com 1 fold e 1 semente
o `unit_id` **e** o `config_id` e todo nome de arquivo continua igual — um
diretorio de run antigo segue legivel e retomavel.

**A semente passou a depender da repeticao, nunca da config.** Era
`base_seed + i`: duas configs diferiam nos hiperparametros **e** no sorteio que
as inicializou, entao parte de toda comparacao era sorte, sem como saber
quanta. Agora configs dentro de uma repeticao partem do mesmo sorteio, e o
espalhamento **entre** repeticoes mede a sorte diretamente.

### `run_cnn_resample()`: o fold e o laco EXTERNO

O cache do fold e o objeto caro (escalonamento ajustado no treino daquele fold,
broadcast sobre todos os patches); treinar uma config sao minutos. Fold por
fora, configs e sementes por dentro: cada cache e construido uma vez e
amortizado sobre o grid. A ordem intuitiva — "para cada config, para cada
fold" — reconstruiria cada cache k vezes a toa.

Tudo cai em **um** diretorio de run: o fold e coluna, nao pasta. E o que
permite agregar sobre folds e sementes sem ninguem costurar diretorio depois.

### Duas tabelas, porque sao perguntas diferentes

| arquivo | pergunta |
|---|---|
| `comparison_ranked.csv` | toda unidade, como foi medida — trilha de auditoria |
| `comparison_by_config.csv` | uma linha por config, media ± sd — a decisao |

Ranquear **unidades** deixaria uma semente sortuda de uma config mediocre
passar na frente da media firme de uma boa. O teste demonstra isso em quatro
linhas de fixture: a config que detem o melhor run isolado nao e a de melhor
media.

`seed_noise_floor()` mede quanto a metrica se move quando **so** a semente
muda. Diferenca entre configs menor que isso nao e evidencia — e a razao de o
`one_se()` da etapa 5 existir.

### Nao copiei o `trainControl()` do caret

Avaliei e descartei. No caret o controle e diferido porque `train()` e quem tem
os dados; aqui o usuario ja tem `meta` carregado quando configura, entao adiar
so criaria um segundo objeto para manter em sincronia, e atrasaria os erros de
plano (k grande demais, buffer que esvazia um fold) para cinco horas depois.
Construtor direto permite `print(plan)` e **ver os folds antes de gastar CPU**.
O que foi copiado do caret e o `resamples()`: media e desvio sobre repeticoes.

### Dois avisos que viraram assercao

`agg$n_failed` numa tibble sem essa coluna avisa antes de devolver NULL; e
`range()` de vetor vazio devolve `-Inf/Inf` com aviso — alcancado mesmo sem
grupo algum, porque o dplyr avalia cada expressao uma vez numa fatia vazia para
inferir o tipo. Os dois corrigidos na causa, e `aggregation_emits_no_warnings`
impede a volta: aviso em funcao de relatorio treina quem usa a ignorar avisos,
e o aviso que importa e sempre o proximo.

---

## 2026-09-13 — etapa 4, consumo: 04 e 99 aprendem a ler unidades

Trocar a unidade de trabalho quebra em silencio quem lia a tabela antiga. Dois
consumidores estavam errados e nenhum dos dois daria erro:

### `04_final_model.R` selecionava a unidade sortuda

Ele fazia `filter(ranking, rank == 1L)$config_id` sobre `comparison_ranked.csv`
— que agora tem uma linha por **unidade**. Selecionar o rank 1 dali e escolher
o melhor **run isolado**, exatamente o erro que repeticao existe para evitar.
Passou a ler `comparison_by_config.csv` (media ± sd), com fallback para a
tabela antiga quando o arquivo nao existe — em run de 1 fold e 1 semente as
duas coincidem de qualquer jeito.

E ganhou o aviso que importa no momento em que a escolha e feita: **se a
vantagem do 1o sobre o 2o for menor que o sd tipico entre sementes**, ele diz
em voz alta que as duas configs sao indistinguiveis com o numero de repeticoes
daquele run. Selecionar assim mesmo e legitimo; nao saber nao e.

### `99` procurava checkpoint por `config_id`

`paste0(comparison$config_id, "_best.pt")` acharia um arquivo que nao existe
mais e, pior, deixaria de checar os que existem — um FAIL falso somado a uma
cobertura perdida. Agora e por `unit_id`, e o mesmo vale para os
`gate_summary.csv`.

Runs anteriores a reamostragem nao tem `unit_id`/`fold`/`seed`. Em vez de dois
caminhos no bloco inteiro, as colunas sao preenchidas com o que aquelas linhas
de fato eram (`unit_id = config_id`, `fold = 1`), e o resto do codigo nao
precisa saber a diferenca.

### Tres checagens novas na etapa 03

| checagem | por que |
|---|---|
| plano de reamostragem gravado com o run | resultado cujos folds nao podem ser reconstruidos e resultado que nao pode ser defendido |
| unidades = configs × folds × sementes | um buraco aqui nao aparece em lugar nenhum: a media de uma config sai de menos repeticoes que as outras e a comparacao fica torta |
| mesmo conjunto de sementes em todas as configs | e o que faz duas configs serem comparadas sob o mesmo sorteio; se cada uma tiver a sua, parte de toda diferenca medida e sorte e nada avisa |

E a que decide se o tuning significou alguma coisa: **o vencedor se destaca
acima do ruido de semente?** WARN quando a distancia entre 1o e 2o e menor que
o sd tipico entre sementes.

---

## 2026-09-13 — `test_resample_run.R`: o teste que prova a FIACAO

Todo outro teste em `tests/` checa **uma funcao**. Este checa que elas estao
**ligadas**: patch store sintetico em disco, plano de fold real, modelos torch
treinando de verdade, e as tabelas de comparacao saindo do outro lado. Existe
porque `run_cnn_resample()` e chamado uma vez, por um script que custa horas —
e os defeitos dessa camada sobrevivem a todos os testes unitarios.

Ele se pagou na primeira execucao, quatro vezes.

### 1. `bind_rows` com tipo adivinhado — bug PRE-EXISTENTE

```
Can't combine `..1$window_sizes` <double> and `..2$window_sizes` <character>
```

`read_csv2()` adivinha o tipo. `window_sizes` vale `"3"` numa config de janela
unica, e ele le como **numero**. Ao empilhar a linha nova (texto) com a relida
(double), o `bind_rows` aborta — **depois de treinar**, perdendo o trabalho da
unidade corrente.

Nao veio da etapa 4: atingia **qualquer retomada** de um grid com config de
janela unica, e o grid do `03` tem tres delas (`c(3)`, `c(9)`, `c(15)`). Com
multi-fold passou a ser garantido, porque o fold 2 rele o que o fold 1
escreveu. Corrigido forcando as colunas de rotulo a `character` na releitura.

**Mesma familia do snapshot com separador decimal**: quem escreve sabe o tipo,
quem rele esta adivinhando. Terceira vez que este projeto e mordido por isso.

### 2. Piso de ruido igual a ZERO com uma semente so

Com `n_seeds = 1` nao existe repeticao — e o relatorio anunciava
`sd mediano: 0`, `estimado em 2 combinacoes`. **Zero e o pior valor possivel**:
faz qualquer diferenca entre configs parecer evidencia.

Duas correcoes, mesma causa — eu contava **linhas** em vez de contar o que de
fato varia:

- exige **valores nao-NA**, nao linhas (duas repeticoes com metrica NA nao dao
  espalhamento nenhum, e o relatorio anunciava cobertura com `sd = NA`);
- exige **sementes distintas** (duas linhas com a mesma semente sao a mesma
  coisa contada duas vezes).

### 3. Nome de unidade abreviado era uma armadilha — decisao minha, revertida

Eu tinha feito `unit_id == config_id` quando havia 1 fold e 1 semente, para
preservar os nomes de arquivo de antes da reamostragem. Consequencia: **mudar
`n_seeds` renomeava as unidades**, entao retomar um run de 1 semente pedindo 3
retreinava tudo e deixava linhas duplicadas na comparacao.

A compatibilidade que isso comprava era com o vazio — a etapa 03 nunca rodou
desde a limpeza das saidas. Nome agora e sempre `{config}_f{fold}_s{semente}`,
o que torna **aumentar `n_seeds` uma mudanca retomavel**: da para comecar com
1, ver o grid de pe, e subir para 3 sem perder nada.

### 4. O teste retomava a propria execucao anterior

`tempdir()` sobrevive entre `source()` na mesma sessao. Quando uma execucao
abortava antes do `unlink` final, a seguinte **retomava os modelos velhos** em
vez de treinar — o resume funcionando exatamente como deveria, contra o teste.
Quatro assercoes falharam medindo modelos de outra rodada. Limpeza passou para
a **entrada**: um teste que retoma o proprio run anterior nao e reproduzivel.

### O mecanismo funcionando, em dados onde a resposta e conhecida

```
cfg_002  CCC 0.473 +/- 0.018      piso de ruido: sd 0.0636 | amplitude 0.1543
cfg_001  CCC 0.468 +/- 0.109
```

Distancia entre as duas: **0,005**. Piso de ruido: **0,064**, treze vezes
maior. As configs sao indistinguiveis, e o relatorio diz isso em vez de coroar
a cfg_002.

### Custo: a suite passou de 3,3 s para 2,8 min

Praticamente tudo no smoke test, o que muda o laco de feedback usado a cada
edicao. Em vez de esconder isso, `run_all.R` separa **rapidos** (~3 s, rode
sem pensar) de **lentos** (~3 min), com `run_slow <- TRUE` no topo. O default
e rodar tudo: sao os unicos testes que provam ligacao entre modulos, e o lugar
onde se pagam e antes de um script caro.

---

## 2026-09-14 — tipo adivinhado na retomada: o conserto certo, na terceira vez

O run espacial (k=3, 3 sementes) morreu no fold 2:

```
Can't combine `..1$weight_decay` <character> and `..2$weight_decay` <double>
dplyr::bind_rows(comparison, row) at train_cnn.R:719
```

Terceira vez que este projeto e mordido pelo mesmo defeito, e a segunda com o
mesmo sintoma. As tres:

| onde | o que o leitor adivinhou |
|---|---|
| snapshot do 99 | `31.190645` -> o PONTO virou separador de milhar |
| `window_sizes` | `"3"` (janela unica) -> lido como NUMERO |
| `weight_decay` | `"1e-04"` -> com decimal virgula nao parseia, virou TEXTO |

### Por que o conserto anterior nao bastou

Eu tinha forcado uma **lista** de colunas a `character` na releitura. Estrategia
errada: a lista nunca esta completa. `weight_decay` nem e coluna de rotulo — e
numerica, e quebrou pelo motivo oposto (notacao cientifica que o locale nao
parseia). Cada coluna nova e outra chance de repetir, e o preco e sempre o
mesmo: o erro acontece **depois** de treinar, perdendo a unidade corrente e
abortando o run.

### O conserto

**A retomada le o RDS, nunca o CSV.** O CSV continua sendo gravado — e a copia
legivel, a trilha de auditoria. O RDS e a copia autoritativa, gravada junto na
mesma funcao (`write_comparison()`) para nunca discordarem. Um RDS guarda a
tibble como ela e: nenhuma coluna para lembrar, nenhum tipo para adivinhar.

O fallback para o CSV fica so para runs gravados antes disto.

**A regra geral, que vale para o resto do pipeline:** CSV e formato de
apresentacao. Quando um dado e escrito pelo codigo para ser lido pelo codigo,
o formato tem que carregar o tipo.

### A assercao

`partial_resume_preserves_every_type` compara o conjunto **inteiro** de classes
antes e depois da retomada parcial, em vez de checar as colunas que eu lembrei
de listar — que foi exatamente o modo como o conserto anterior falhou. E o grid
do smoke test ganhou `weight_decay = 1e-4` de proposito: com um valor "redondo"
o teste atravessaria o bug sem ve-lo.

### Nota sobre o run perdido

O fold 1 inteiro (9 unidades) sobreviveu: checkpoints e linhas gravados. A
unidade `cfg_001_f2_s1` tem checkpoint mas nao tem linha na comparacao, entao a
retomada a retreina — o criterio de "pronto" exige **as duas** coisas, que e o
que impede tratar um treino interrompido como concluido. Custo do bug: uma
unidade.

---

## 2026-09-14 — revisao de concepcao e prospeccao

Auditoria desde a concepcao + prospeccao de rumos, em
[`docs/revisao_e_prospeccao_2026_09.md`](revisao_e_prospeccao_2026_09.md).
Nada executado — e material de decisao.

Os tres achados que mudam o que fazer a seguir:

1. **O conjunto de TESTE continua sendo um split aleatorio.** A etapa 4 limpou
   a validacao (0% por fold), mas 28,08% dos pontos de teste ainda dividem
   pixel com treino, e 55% dividem pixels de patch na janela 15. Toda metrica
   `test_*` de hoje e otimista, e nada avisa no momento em que ela e lida.

2. **As tres funcoes sem teste sao as tres que mentem em silencio:**
   `spatial_overlap_report()` (sub-reportar = 0% falso),
   `calc_metrics()` (erro de formula = ranking errado),
   `compare_run_snapshot()` ("tudo identico" falso). O pipeline ja **afirma**
   0% de vazamento por fold com base numa funcao nunca verificada contra um
   caso de resposta conhecida.

3. **A regra de buffer que escrevi ontem esta ERRADA por geometria.**
   Reimplementei o relatorio de vazamento em Python, direto dos arquivos
   brutos, e comparei. O 0% de "mesmo pixel" e verdadeiro e confirmado — mas
   0,51% a 0,98% da validacao por fold ainda divide **pixels de patch** com o
   treino. Causa: sobreposicao de patch e condicao de **quadrado**
   (`|dlinha| <= w-1` E `|dcoluna| <= w-1`); `apply_buffer()` mede distancia
   **euclidiana**, um circulo. Dois centros a 14 linhas E 14 colunas distam
   14*sqrt(2) ~ 19,8 celulas — fora do circulo de 15 — e os patches ainda
   dividem o pixel do canto.

   Correcao certa: `metric = "chebyshev"` no buffer (default), que e a
   geometria dos patches e faz `buffer = w * res` voltar a ser exato. A
   alternativa preguicosa e multiplicar por sqrt(2), que funciona e descarta
   41% mais treino que o necessario.

   **Nao apliquei** — o run estava em andamento. Corrigi so o comentario em
   `R/resample.R`, que afirmava a regra errada; comportamento inalterado.
   Gravidade media: contra 54,2% do split fixo, 0,5-1% e melhora de quase duas
   ordens de grandeza. Mas e a diferenca entre "praticamente separado" e
   "separado", e a segunda e a que estava escrita como garantia.

   **Como passou:** o `03` imprime so a linha `same raster cell`. O
   `fold_leakage_report()` calcula a sobreposicao por janela — o numero estava
   la, filtrado fora da tela.

4. **`padding = "valid"` e uma decisao que expira.** A predicao totalmente
   convolucional seria ~w² (225x na janela 15) mais rapida, mas so e
   **exata** se a rede for treinada sem zero-padding — com `padding = 1` as
   posicoes de borda do patch veem zeros no treino e veriam vizinhos reais na
   FCN, e tanto `flatten` quanto `gap` consomem a borda. Eu ia recomendar a
   FCN sem ressalva; a conferencia derrubou metade da ideia. Decidir **antes**
   do treino final: depois, significa retreinar tudo.

Tambem registrado o que foi avaliado e **nao** recomendado, com motivo:
adotar `luz`, integrar com `tidymodels`/`parsnip`, trocar `dplyr` por
`data.table`, paralelizar configs, comprimir o store.

---

## 2026-09-14 — primeiro run espacial completo: o otimismo medido

27 unidades (3 configs x 3 folds x 3 sementes), `spatial_folds(k=3,
block_size=2, buffer=15px)`, nenhuma falha. `99`: PASS 60 / WARN 3 / FAIL 0.

### O resultado que vale o run inteiro

| config | val_ccc (espacial) | test_ccc (aleatorio) | diferenca |
|---|---|---|---|
| cfg_003 | 0,5257 | 0,5613 | +0,036 |
| cfg_001 | 0,5170 | 0,5661 | +0,049 |
| cfg_002 | 0,5071 | 0,5478 | +0,041 |
| | | **media** | **+0,042** |

**O conjunto de teste e mais FACIL que a validacao**, nas tres configs. E o
contrario do que se espera, e e a confirmacao empirica direta do §2.1 da
revisao: a validacao agora e espacialmente limpa (0% de mesmo pixel por fold),
o teste continua sendo split aleatorio com 28% de mesmo pixel e 55% de patch
sobreposto.

**+0,042 de CCC e o preco da vizinhanca**, medido nos proprios dados. Primeira
vez que este projeto quantifica isso.

### O grid nao separou, e o pipeline disse

```
distancia 1o-2o : 0,0086
piso de ruido   : 0,0275   (3,2x MAIOR)
```

Aplicando a regra `one_se` da etapa 5: limiar = 0,5257 - 0,0084 = 0,51731; a
cfg_001 fica em 0,51704, **fora por 0,0003**. Nem a regra robusta decide — ela
erra o empate por tres decimos de milesimo. Com `tune_length = 3` nao ha busca,
ha tres sorteios; e cfg_002/cfg_003 sao **a mesma arquitetura** variando so lr
e weight decay.

### Decomposicao da variancia — achado inesperado

```
sd das 27 unidades              : 0,0253
sd entre medias de FOLD         : 0,0086
sd entre medias de SEMENTE      : 0,0081
sd entre sementes no mesmo fold : 0,0275   <- domina
```

**Os tres folds espaciais concordam entre si mais do que sementes repetidas do
mesmo fold concordam.** Amplitude entre folds: 0,0165, menor que o ruido de uma
semente.

Bom sinal para a construcao dos blocos: nenhuma regiao e dramaticamente mais
dificil na escala de 2 graus. Sinal ruim para o treino: **a inicializacao dos
pesos importa mais que a geografia**. Combina com o early stopping disparando
na epoca 6-21 — o modelo esgota o que os rotulos tem a dizer rapido demais, e o
que sobra e sorte. E o argumento mais forte a favor do pre-treino
auto-supervisionado (§D4 da revisao).

### Tres defeitos de RELATORIO corrigidos

Nenhum muda resultado; todos faziam o pipeline dizer uma coisa por outra:

1. `Resume: 18/9 units already trained` — numerador contava o run inteiro,
   denominador so o fold. Agora reporta os dois separadamente.
2. `todas as configs do grid tem linha na comparacao | 27/3 configs` —
   numerador em unidades, denominador em configs. Agora `3/3 configs (27
   unidades)`.
3. **O mais serio:** o WARN do vencedor dizia `sd tipico entre sementes =
   0,0251`, mas 0,0251 e o sd de TODAS as unidades da config (mistura fold e
   semente). O sd entre sementes e 0,0275, que e o que o `03` imprime. Duas
   quantidades diferentes com o mesmo rotulo, em dois lugares do mesmo
   relatorio. O `99` passou a chamar `seed_noise_floor()`, a mesma fonte do
   `03`.

---

## Phase 2 — the store lock (2026-09-14)

Written while stage 01 was running in dev mode, because nothing in phases 2-4
touches stage 01.

### The problem it solves

A patch store carries no visible mark of the predictor set, the target or the
resolution it was built under. Read it with a script expecting something else
and **nothing errors**: the network trains, converges, and produces a map of
the wrong variable. This is the shape of every restart this project has had --
the defect lived upstream of where it was found.

### What changed

**`02_extract_patches.R`** records the spec in the manifest: `target_col`,
`target_transform`, `cell_size`. It now reads `target_config.csv` for the
target name -- the only reason it reads that file at all.

The predictor LIST is recorded, not a hash of it. Comparing lists costs the
same and lets the error name *which* predictors differ. A hash can only say
"different".

**`R/dataset.R`** gained `store_spec()` and `check_store_spec()`.
`store_spec()` reads the windows from `manifest$windows_extracted`, **not**
from `store$window_sizes` -- the latter is the `window_sizes =` argument, so
asking the store what it holds through that field only echoes the question
back. This was a real bug in the first draft.

`check_store_spec()` reports **every** mismatch, not the first: discovering
one, fixing it, re-running and discovering the next is the slow way to learn
there were three. `strict = FALSE` returns the messages instead of stopping,
which is what a reporting script wants.

It also refuses a **reordered** predictor set whose members are identical. The
order is the contract tying channel *i* to band *i*; scrambled, the network is
fed one predictor while the map is built from another, with no error and
plausible metrics.

A store written before the spec existed records `NA` and is **tolerated**, not
failed -- "cannot answer" is not "answered wrongly", and failing it would force
a re-extraction to gain information the store already implies.

**`03_run_tuning.R`** calls it right after `cell_size` is read, before anything
expensive. One file read; refuses in seconds what would waste hours.

**`99_check_pipeline.R`** compares the recorded target against `target_config`
and the recorded `cell_size` against the rasters as they are *today*.

### Discarded

- **A hash of the whole configuration.** Cheap to compare, useless to read: it
  can report a mismatch but never which field, and the field is the fix.
- **Failing an old store.** Rejected above.
- **Checking in `load_patch_store()`.** The loader does not know the grid, the
  target or the resolution the *run* wants -- only stage 03 does. Putting the
  check there would mean either passing all three into the loader, or checking
  a weaker thing in a place that looks authoritative.

### Also fixed, found on the way

**The hardcoded `windows_extracted == "3, 9, 15"` in the 99.** An example value
frozen into a checker a package user cannot change without editing the checker
-- the exact opposite of the requirement. It now checks the *shape* (odd,
ascending), which holds for any store, and leaves "can this store serve this
grid" to `check_store_spec()`, in stage 03, where the grid exists.

**A live bug in `05_predict_spatial.R`, mine, from the scaling refactor.**
`predictor_scaling_file` was built from `final_run_dir` and `config_id` about
fifteen lines *before* either was resolved -- both can be `"latest"`/`"auto"`.
Moved to after the resolution.

---

## `05` can predict on another grid (2026-09-14)

Not in the numbered plan, and the end-to-end dev run does not exist without it:
predicting the 250 m grid is measured in days.

`predict_raster_dir` (env var `soc_predict_raster_dir`) remaps the raster paths
by **file name**, keeping the training channel ORDER exactly -- the
alphabetical order a directory listing returns is not the contract. A missing
predictor aborts: the network has a weight for every one of them.

**It does not resample, and that is stated loudly at runtime.** The window is
counted in PIXELS, so the 15 x 15 patch that spans 3.75 km at 250 m spans
300 km at 20 km. The network is shown a neighbourhood it was never trained on.
Such a run proves the **wiring**; it never proves the map. The script compares
the prediction resolution against the `cell_size` the store recorded -- the
manifest lock paying for itself immediately -- and prints the ratio in a banner
rather than leaving it to be noticed.

### Tests

`tests/test_store_spec.R`, 21 assertions, no torch, registered in `run_all.R`.
Mostly refusals: a lock is worth only what it refuses, and a broken one takes
exactly the shape of a lock that passes everything.

---

## Phase 3 — the registry, and the baselines that make it necessary (2026-09-14)

### The problem

"The CNN reached CCC 0.62" is a number with no scale on it. The project's
central claim -- that a convolution over a neighbourhood beats the same
predictors read at a point -- had never been measured against anything.

### Four models, one fold plan

| family | input | answers |
|---|---|---|
| `rf_centre` | centre pixel | the classic DSM baseline |
| `rf_context` | centre + per-channel window means | context **without** spatial structure |
| `mlp_centre` | centre pixel | the architecture, or just the covariates? |
| `cnn` | the whole patch | context **with** spatial structure |

**The gap between `rf_context` and the CNN is what the convolution is worth.**
If they match, the convolution is doing averaging -- falsifiable, cheap, and
measured under folds identical by construction.

### What was built

**`R/model_registry.R`** -- `model_spec()`, `register_model()`, `get_model()`,
`list_models()`. Five fields: `name`, `input`, `fit`, `predict`,
`default_grid`, `count_params`. `input` is the one that matters: it says which
VIEW of the fold cache a model consumes, so adding a model never means editing
a runner.

The contract is checked at **declaration** time, not at fit time. Minutes into
a fold is the wrong moment to learn an argument is named `data` instead of `x`.

`register_model()` refuses a silent overwrite. Two different models answering
to one name in one session produce results carrying no mark of which ran.

**Random Forest forced the design to be honest.** An interface designed against
one implementation only ever describes that implementation; RF has no epochs,
no learning rate, no device, and consumes a matrix rather than a 4-D tensor.

**`fold_table_view()` in `R/dataset.R`** derives the table from the tensors the
fold cache already holds -- same training rows, same scaling fitted on them,
same buffer. A separate tabular extraction would be a second code path
producing numbers that only look like the first one's.

It **verifies** that the centre pixel agrees across windows rather than
assuming it. Concentric patches around one point must share their centre; if
they do not, every table feature describes a different location than the
tensors do, and no metric would reveal it.

A 1x1 window contributes no window-mean column: its mean *is* its centre, and a
duplicated column is one a tree can split on twice for free.

**`R/train_table.R`** -- `run_table_resample()`, same fold loop, same seed
discipline, **same comparison table shape**, so `summarise_resamples()`,
`seed_noise_floor()` and `one_se()` work unchanged across families. That shared
shape is the only thing the two runners need to share.

**`examples/soc_stock_0_5cm/03b_run_baselines.R`** reads `fold_plan.rds` from
the CNN run rather than rebuilding the plan. Rebuilding it "the same way" makes
the comparison depend on two call sites staying in step; reading it makes them
identical by construction. It runs the same `n_seeds` as the CNN -- a baseline
with one seed against a CNN with three reads as if only one of them were
uncertain -- and prints the gap **next to the noise floor**.

### Discarded

- **Generalising `run_cnn_resample()` to take a model_spec.** The CNN path
  carries epoch histories, gate analyses, per-quantile metrics, checkpoints,
  DataLoaders and a device, none of which a forest has; every one would become
  an `if` inside the one function this project cannot afford to destabilise.
  Every restart here has had the same cause -- a defect introduced upstream of
  where it was noticed -- and rewriting the trained path days before the
  definitive run is exactly that move. If the two ever converge on what they
  need, they can be merged then, against two real implementations instead of
  one imagined interface.
- **Caret's `modelInfo`.** It carries `library`, `type`, `sort`, `loop`,
  `levels`, `oob`, `varImp` because it supports 200+ models. Fields get added
  here when a model needs one.
- **Tuning `n_trees`.** More trees never overfit a forest and the curve is flat
  long before 500; tuning it spends budget on the parameter with a known answer.
- **A separate tabular extraction from the rasters.** Rejected above.

### `ranger` is not installed here

The RF baseline uses `ranger` when present and falls back to `randomForest`.
They are not equivalent in cost: on 30k rows and ~360 features `randomForest`
is single-threaded and slow enough to dominate a run that also trains neural
networks. The fallback exists so the baseline RUNS, and the comparison table
records which backend produced it.

    install.packages("ranger")

### Tests

`tests/test_model_registry.R`, registered in `run_all.R`. The fake cache is
filled so that the centre and the mean are two DIFFERENT known numbers -- a
function that confuses them cannot pass both. It also checks that a reordered
prediction table is refused: a forest reads features by position, and that is
the case that produces a confident answer from the wrong covariates.

---

## Borrowing caret's model library (2026-09-14)

Cassio asked whether the extra families should come from caret rather than be
hand-written, so caret does the heavy lifting as more models are added.

**Yes -- as an adapter, not as the foundation.**

### What caret is worth here

Its `modelInfo` objects carry, for ~230 methods, the parameter NAMES, a grid
generator that knows sensible ranges, the fit and predict closures, and which
package to load. That is the expensive part of supporting many models, and
re-deriving it once per family is how ranges quietly end up slightly wrong.

### What caret does not get to do: resample

The fold plan carves the test set and the folds by one criterion, with block
structure and a Chebyshev buffer. `trainControl` *can* express arbitrary folds
through `index`/`indexOut`, so this is not impossible -- it is undesirable. It
would mean translating our plan into caret's index lists, reading our metrics
back through a `summaryFunction`, and having two objects that each believe they
own the resampling. `seed_noise_floor()` and `one_se()` are already ours and
already tested.

So: `trainControl(method = "none")` with a one-row `tuneGrid`. caret is a
fit/predict adapter; our loop stays in charge of folds, seeds, scaling and
metrics.

### Three settings that are not negotiable

| setting | why |
|---|---|
| `returnData = FALSE` | `train()` otherwise stores the training data in the fitted object: ~85 MB per unit at 30k x 360, and a run holds many units |
| no `preProcess` | the fold cache is already scaled on this fold's training rows; a second scaling is harmless until someone changes one of the two, and then the network and the baseline are standardised differently while every table still says they were compared |
| `allowParallel = FALSE` | the parallelism belongs to the model (ranger's threads, xgboost's nthread); nesting a foreach backend oversubscribes the cores |

### The one thing that forced a contract change

caret's grid generators need the **real** training data. Some use only
`ncol(x)` (mtry), but `glmnet` computes its lambda path from the VALUES. A grid
built against a synthetic matrix of the right width is correct for the first
kind and quietly wrong for the second.

So `default_grid` became `function(tune_length, seed, x, y)`, and
`run_table_resample()` generates the grid **inside fold 1**, where the table
exists -- once, then reused for every fold after it. A grid regenerated per
fold would tune a different set of configs on each one, and the per-config
means every decision is read from would average over configs that are not the
same config.

A generator that needs neither argument still has to accept them: a signature
that varies per model is one the runner cannot call.

### caret is a SUGGESTS, never a dependency

caret Depends on ggplot2 and lattice and Imports recipes, plyr, pROC,
ModelMetrics, reshape2, foreach -- a heavy tree for someone installing this to
fit a CNN. `R/caret_adapter.R` sources like any other file and nothing fails
until `caret_spec()` is actually called.

### What stays hand-written, and why

`rf` (ranger's native API is faster than going through caret) and `mlp` (it has
to use the CNN's own training recipe, or the control answers nothing). Adding
anything else is now three lines:

```r
register_model(caret_spec("xgbTree"), overwrite = TRUE)
results$xgb <- run_table_resample(model = "xgbTree", ...)   # tune_grid = NULL
```

`caret_available("boost|forest|svm|glmnet")` lists what is on offer. The
model's own package must be installed -- caret Suggests them.

---

## Phase 4 — the cheap questions (2026-09-14)

### 4.2 `conv_padding`: what the border of a patch is worth

`padding = 1` keeps the spatial size by inventing a ring of zeros.
`padding = 0` uses only measured values and shrinks by 2 per 3x3 block.

The argument for valid is sharper here than it looks. With window 3 and two
conv blocks the centre's receptive field is already 5x5 -- **larger than the
patch** -- so under "same" every output position depends on invented zeros, and
the network spends capacity learning the shape of its own border.

The argument against is that it shrinks, and `w - 2b < 1` is not a model. So
this is a **per-branch** decision, and the grid carries three values:

| value | meaning |
|---|---|
| `same` | the previous behaviour, and still the default |
| `valid` | no padding anywhere; needs `window > 2 x blocks` |
| `valid_large` | valid on the LARGE branch, same on the small |

`valid_large` is the useful one. A 15x15 patch re-reads each pixel 225 times
across the dataset; a 3x3 patch only 9. Trading border pixels for honest ones
is nearly free on the first and expensive on the second.

**Three places this could have gone wrong quietly, and what was done:**

1. `embed_in_size` is computed from `out_size`, not `window_size`. Sizing the
   linear layer from the patch is a shape error raised on the first forward
   pass -- after the fold cache has been built.
2. A residual block under valid has a main path **smaller** than its skip. A
   1x1 conv fixes channels, never the spatial size, so the skip is centre
   cropped in `forward()`. Without the crop the addition either errors or, if
   the sizes happened to align, adds the wrong pixels to the wrong place.
3. A padding the geometry cannot honour is normalised **in the grid**
   (`.normalise_conv_padding()`), not only in the model. The model resolves
   `valid_large` to `same` where a branch cannot shrink; left unnormalised, two
   grid rows that build the IDENTICAL model survive de-duplication under
   different names, and the grid trains the same thing twice and reports it as
   two results.

A branch asked for a padding it cannot produce **stops at construction** and
names the constraint, rather than failing inside torch with a shape error that
names no cause.

Backwards compatible: a config row without `conv_padding` builds with `same` --
what that row meant when it was written.

### 4.1 `early_stopping_bias()` -- the tool, not yet the answer

Early stopping picks the epoch with the lowest validation loss and the metric
is read at that epoch, from the same data that chose it. The reported number is
the minimum of a noisy sequence, which is below its mean by construction.

The estimator treats the epochs in the plateau around the chosen one as
exchangeable and takes `mean(plateau) - min(plateau)`. It reads histories
already on disk; **nothing is retrained**.

Its limits are stated in the code and in the printed verdict: it is in LOSS
units, not CCC; it assumes the plateau has flattened, so it is an upper bound
whenever training had not converged -- and the function reports what fraction
of units were still descending, so that case is visible rather than silently
folded into the number.

What it decides: if the bias is small, no third split, and 15% of the data
stays in **training**. If it is large, part of each config's score is the luck
of its best epoch, and the real experiment is worth running.

Needs a dev run's histories. The full-data histories that would have answered
it were deleted with the rest of the outputs; only the summary in
`reference_performance.md` survives.

### 4.3 kNNDM: the finding is about the CRS, not the cost

`_measure_knndm.R` measures rather than assumes. The important finding does not
need CAST installed (it is not, here):

**The points are global lon/lat, and kNNDM must not be run on them directly.**
A degree of longitude is 111 km at the equator and 0 at the pole; a distance
computed over that is not a distance. Projecting to an equal-area CRS
(Mollweide) is correct regardless of cost.

It also happens to settle the cost question. Unprojected, nearest-neighbour
search needs full pairwise geodesic distances: at n = 31,000 that is 9.6e8
pairs, ~7.7 GB for one double matrix. Projected, a kd-tree does it in
O(n log n) and the script measures the actual growth exponent rather than
trusting the theory.

What this does **not** settle: whether kNNDM folds are better than the block
folds in use. That is empirical and costs a full run, which is not being spent
before the pipeline is stable.

---

## `width = Inf` is a tibble feature (2026-09-14)

The first dev run of stage 01 died at the very last print before the export:

    Error in print.default(...) : invalid printing width
    NAs introduzidos por coercao a intervalo de inteiros

`print(x, width = Inf)` is dispatched by `print.tbl_df`. On a plain
`data.frame` it reaches `print.data.frame` -> `print.default`, where `width` is
coerced to integer, `Inf` becomes `NA`, and the call dies naming nothing that
points at the cause.

`dataset_check` is built by `summarise()` over `dataset_model_split`, which is
a data.frame because `terra::extract()` returns one. So the summary was a
data.frame and the script stopped **after the expensive extraction and before
writing anything** -- the worst possible place.

### Fixed as a class, not as a line

`print_wide()` in `R/utils.R` coerces to a tibble and then prints. Every
`print(..., width = Inf)` call site in `R/` and `examples/` now goes through it
-- there were 14, and each was the same latent failure waiting for a
data.frame.

"Remember to pass a tibble" is not a contract a package can rely on: anyone
using this framework on their own data.frames hits the identical wall.

---

## A constant channel means two different things (2026-09-14)

The first dev run of stage 01 flagged 7 constant channels that survived the
drop list: `change_lulc_*_ocean`, `*_other_change`, `*_stable_natural`,
`geology_evaporites`, `soil_class_fao_anthrosols`, `soil_class_fao_glaciers`,
`terrestrial_habitat_marine_intertidal`.

Both the script and the 99 told the user to add them to
`manual_predictor_drop`. **That advice was wrong under a dev profile**, and
following it would have been expensive in a way that takes weeks to notice.

### Why

At full size, constant is a statement about the DATA: the channel carries no
information at any profile while being non-zero somewhere on the map, so its
weights never receive a gradient and stay at random init exactly where the
network extrapolates. Real defect.

At 10%, it is usually a statement about the DRAW. Every one of those seven is a
rare class -- glaciers, evaporites, marine intertidal -- all-zero at 3,766
points because the subsample missed the handful of profiles that carry it.

Dropping them on that evidence changes the predictor set of the **full** run
from an artefact of a 10% draw. And the predictor set is one of the three
things a patch store is locked to (Phase 2), so the dev store and the full
store would become incompatible by construction, for no reason at all.

**The rule: a channel is a drop candidate when it is constant at FULL size.**

Both the stage 01 warning and the 99 check are now profile-aware. The 99 FAILs
at `full` and WARNs at `dev`, saying why -- a false FAIL during the runs that
are cheapest to do is how people learn to ignore a checker.

### The channel that actually stops a run is a continuous one

Following this up found the real risk, which nothing was checking.

`build_fold_cache()` refuses to z-score a channel whose sd is zero over a
fold's TRAINING rows, and refuses by stopping. Dummies are safe --
`fit_scaling()` gives them centre 0, scale 1, and never divides by their
spread, which is why the seven above will not stop anything. A **continuous**
channel with almost no spread is a different story: it passes every global
check, then goes flat inside one fold and stops stage 03 partway through, after
the patches have been extracted.

The 99 now measures sd/mean per continuous channel and FAILs on anything below
1e-6, naming the channels. Relative to the mean because the units are wildly
different: sd = 0.001 is nothing for elevation and everything for a vegetation
index.

Cheap to see between 01 and 02; expensive to meet in the middle of 03.

---

## One bad escape stopped the 99 before its first check (2026-09-14)

    Erro: '\.' is an unrecognized escape in character string
    (R/diagnostics.R:351:56)

`early_stopping_bias()` was appended to `R/diagnostics.R` through a shell
heredoc, which collapsed `\.` to `\.`. R does not compile a file lazily: a
single invalid escape makes the WHOLE file unparseable, so one character took
down every function in `diagnostics.R` -- and the 99 refused to start, one line
into a check that had nothing to do with it.

This is the second time a heredoc has corrupted a file this way (the first
turned `
` into a real newline). The tool is the problem, not the typo:
**write a script file and run it, never a heredoc containing backslashes.**

### The guard, which is what actually matters

`tests/test_sources_parse.R` parses every `.R` file under `R/`, `examples/` and
`tests/`. It runs **first** in `run_all.R`.

`parse()` answers exactly this question and touches nothing -- no package
loaded, no code run, nothing evaluated -- at milliseconds per file. There is no
point testing behaviour in a file that cannot be read, and a syntax error
should not be delivered to whoever next sources the file, minutes into a run,
pointing at a file they were not editing.

It covers `examples/` too: a pipeline script that cannot parse wastes the same
day as a framework file that cannot, and the scripts are what get edited most.

---

## The 99 still referenced two objects the refactor deleted (2026-09-14)

    Error: objeto 'scaling' nao encontrado

`nrow(scaling)` and `sum(split$n)`. Both are valid syntax, so `parse()` cannot
see them; they only fail when the line is reached -- which in a checker is
after the stage it was meant to guard has already run. Stage 01 has not written
`predictor_scaling.csv` since the scaling became a property of the fitted
model, and it has not written a split since the fold plan took that over, so
these lines had been dead for as long as those changes.

### Replaced rather than deleted

Both checks were worth having; they were pointed at the wrong files.

- `n_predictors: raster_table vs predictor_scaling` is gone. Checking for that
  file would now be checking that a file which must NOT exist does.
- `dataset rows vs the sum of the splits` became three checks: dataset vs
  `point_metadata`, vs `qc_summary$n_rows_after_qc`, vs `dataset_check$n_rows`.
  Stage 01 writes those from three different objects in three different blocks,
  so a disagreement means one was built from a stale copy -- which is how a
  store once ended up with more patches than there were points.
- The 70/15/15 proportion check is gone with the split. `fold_sizes.csv` in
  stage 03 is where those proportions get checked now, against the plan that
  will actually run.

### The rest of the pipeline was swept for the same thing

A static scan of every pipeline script for names used but never assigned
returned 14 more hits, and **all 14 were false positives** -- chained access
(`store$meta$sample_id`, `result$gate$summary`, `pl$params$buffer`,
`cmp$diff$status`), which the heuristic reads as a bare name. `scaling` and
`split` were the only real ones.

No permanent automated guard was added for this. `codetools::checkUsage` is the
right tool in principle, but on a SCRIPT every function from a sourced file and
every library export is reported as a global, which is a checker nobody reads.
The guard that works is the one that already caught it: the 99 has to RUN, end
to end, before every expensive stage.

### Also

`rf_spec()` used `cfg$mtry` on a tibble that may not carry the column, which
returns NULL *and* warns. Now `[[ ]]` behind a `names()` check -- the same fix
`build_cnn_from_config()` already carried for `embed_pool`, for the same
reason: the path is deliberately optional, and a warning on it trains people to
ignore warnings.

---

## A 10% subsample does not make stage 02 ten times faster (2026-09-14)

Measured from the dev run's own `point_metadata.csv` before launching 02, so
the expectation would be a number rather than a hope.

Stage 02 costs RASTER READING, not points: one strip per band per chunk, with
empty chunks skipped. A subsample is faster by whatever fraction of the raster
its points happen to MISS -- which depends on how they are spread, not on how
many there are.

The 3,766 dev points span latitude -47 to +70. At `chunk_nrows = 1000` that is
43 of 87 chunks and **50.5%** of the raster rows: roughly half the full run's
reading, not a tenth.

### The lever, and why it was not pulled

| chunk_nrows | chunks with points | % of raster read |
|---|---|---|
| 250 | 112 | 34.2 |
| 500 | 71 | 42.2 |
| **1000 (current)** | **43** | **50.5** |
| 2000 | 25 | 58.3 |

Smaller chunks read fewer rows and issue more GDAL calls. The note in `02`
records that raising 200 -> 1000 was a win precisely because it cut the call
count five-fold -- which makes sense at full density, where almost every chunk
holds a point and shrinking skips nothing.

For a sparse subsample the sign flips. **Which wins here depends on the
per-call overhead, which has never been measured**, so the setting was left
alone rather than gambling a run on a guess.

What was added instead is the number: 02 now prints, before the loop, how many
chunks hold a point, what share of the raster will be read, and how many strip
reads that is. Deciding whether to wait should not require watching progress
lines and extrapolating.

---

## A block size measured on the full data was wrong for the subsample (2026-09-14)

Audited stage 03 while 02 was extracting, because today had already cost two
stops from references that were valid when written.

`03` carried `block_size = 2`, with a table of measurements beside it from the
FULL point set: 1,279 blocks, largest block 4.4% of the points. Correct then.

**On this store the same 2 degrees gives a largest block holding 34% of the
points.** With k = 3, that one block would have decided a fold, and the fold
would have been scored on whatever single landscape it happens to be.

### Why, and why nothing would have said so

Block-subsampling keeps **whole blocks**. A 10% draw therefore has a tenth of
the blocks at the **same width** -- so the block that held 4.4% of 41,385
points holds 34% of 3,766. The point set changed; the constant did not. The
comment beside it even said "re-measure if the point set changes", which is a
note asking a human to remember something, and this is the fifth restart of a
project that keeps being bitten by exactly that.

### What replaced it

`block_share()` and `suggest_block_size()` in `R/resample.R`, and stage 03 now
measures instead of asserting:

```r
block_choice <- suggest_block_size(store$meta, k = k_folds, max_share = 0.10)
print_block_choice(block_choice)
```

It takes the **largest** size whose worst block still fits the constraint --
largest, because separation is what is being bought, and the constraint is the
only reason not to take more of it. On this store it picks **0.25 deg**
(largest block 6.0%, 980 blocks); on the full set the same rule would pick 3.

`spatial_folds()` also **warns** whenever the blocking it is handed lets one
block exceed `1/k` of the points. Warned, not refused: there are point sets
where this is simply true and known, and a refusal would be the framework
overruling a deliberate choice.

### The consequence, stated rather than hidden

0.25 deg (~27 km) separates less than 2 deg (~222 km). **The dev run's spatial
CV is therefore more optimistic than the full run's will be** -- which is
already the rule in `reference_performance.md` (a dev number is comparable only
to another dev number), and is now true for one more reason.

### Tests

The fixture makes the answer known by construction: 300 points inside a 0.4-deg
box plus 100 spread over 20 deg. A large block swallows the cluster whole, a
small one cuts it up, and both monotonicity properties (bigger blocks -> fewer
blocks, lumpier worst case) are asserted. The warning paths are tested too --
including that a balanced blocking stays quiet, which is what makes the noisy
case mean something.

---

# Review night, 2026-09-15

Cassio asked for a review of everything built so far, checked against current
practice, and hardened. Three things came out of it: a leakage mode this
framework was open to, a recommendation from the literature it did not follow,
and a front end that did not exist.

---

## 1. Leave-profile-out: a leakage this framework was open to

**Wang et al. (2025), Geoderma -- "The problematic case of data leakage: a case
for leave-profile-out cross-validation in 3-dimensional digital soil mapping".**

In 3-D soil mapping one profile yields several rows -- 0-5, 5-15, 15-30 cm --
at IDENTICAL coordinates, from the same pit, described by the same surveyor on
the same day. Split those across training and validation and the model is
scored on a depth of a profile it already learned: the covariates are
byte-identical and the target is autocorrelated down the column.

`spatial_folds()` and `region_folds()` happened to be safe, because rows at one
coordinate fall in one block. **`holdout()` and `random_folds()` were not** --
they treated every row as its own unit, which is exactly the failure the paper
describes.

### What changed

`.resolve_row_group()` in `R/resample.R`, and a `group` argument on `holdout()`
and `random_folds()` defaulting to `"auto"`.

**Why the default is `"auto"` and not `NULL`.** Keeping a profile together is
never wrong: with one row per profile it changes nothing at all, and with
several it is the only correct answer. Failing to do it is silently wrong. A
default that is safe in both cases and costs nothing in the common one does not
deserve to be opt-in. It applies the grouping when it MATTERS -- the column
exists and has duplicates -- and says so rather than doing it quietly.

This dataset has 3,766 rows in 3,766 profiles, so nothing changes here. It
changes everything for a package user with depth intervals.

### Proven, not trusted

`check_fold_plan(plan, meta = ...)` now verifies that no group is split across
roles, and **both runners pass `meta`**, so the property is checked against the
data on every run. Four constructors are meant to produce it; "meant to" is the
operative phrase, and the property is what matters, not the arguments intended
to produce it.

---

## 2. Area of applicability: the literature's standing recommendation

**Meyer & Pebesma (2021), Methods in Ecology and Evolution 12:1620-1633.** The
recommendation is not that an AOA is nice to have -- it is that *"the AOA should
be provided alongside the prediction map and complementary to the communication
of validation performances."* We had none.

The problem it solves is specific: a map has a value at every pixel, including
pixels whose predictor combination the model never saw, and nothing in the
raster distinguishes them. The cross-validated CCC printed beside the map does
not apply to the second kind at all.

### `R/aoa.R`

| function | what it does |
|---|---|
| `di_reference()` | summarises the training set: weighted space, mean pairwise distance |
| `dissimilarity_index()` | nearest-training-point distance / that mean, per row |
| `aoa_threshold()` | outlier-removed max of the CROSS-VALIDATED training DI |
| `inside_aoa()` / `print_aoa()` | the mask, and a readable report |

Two decisions worth recording:

**The DI is expressed in units of the training set's own mean pairwise
distance.** That is what makes a DI comparable between datasets instead of
being an arbitrary number of metres, and it is why `test_aoa.R` asserts that
multiplying the entire space by 100 leaves every DI unchanged.

**The threshold comes from ACROSS folds.** For each training point, the
distance to the nearest training point that is NOT in its fold. This is the
whole idea: it asks how dissimilar a point can be and still have been predicted
well during cross-validation, rather than picking a number. A within-fold
version would use each point's own neighbours -- including ones it trained
beside -- and come out far too small. The test fixture makes the two answers
differ by orders of magnitude so an implementation that confuses them cannot
pass.

**Which feature space, for a CNN.** The model consumes patches, so "predictor
space" needs a choice. The CENTRE PIXEL: it is the space a soil scientist
reasons about, it is exactly the space the RF baseline lives in, and it makes
the DI of the CNN's map comparable with the baseline's. A patch distance would
be defensible and is not offered, because a number nobody can interpret is
worse than one that is only mostly right.

**What it is not:** an uncertainty estimate. Inside the AOA means the predictor
combination resembles the training data, not that the prediction is accurate.
Outside, the cross-validation error does not apply, and the honest report is
"not applicable" -- not a wider interval.

---

## 3. The front end

Fitting a model required eight manual steps: ten `source()` lines in an order
that is not guessable, opening the store, reading two CSVs, aligning points to
the store, opening a raster for its resolution, computing a buffer, and copying
a block size. **Three of those have each cost this project a run.**

```r
source("R/load_all.R")

data <- dsm_load(patch_dir, points, type_table, raster_table = ...)
cv   <- spatial_cv(k = 5, block_size = "auto", buffer = "auto")
fit  <- dsm_train(data, model = "cnn", resampling = cv,
                  tune_length = 30, n_seeds = 3, transform = expm1)
```

### What is borrowed from caret, and what is not

Borrowed is the SHAPE: one fitting function, a small object describing the
resampling, a named model, `tune_length` as a budget rather than a lattice.

Not borrowed is `trainControl()`'s thirty arguments. caret carries them because
caret also owns the preprocessing, the parallel backend, the sampling and the
summary functions. Here the spec carries the handful of things that decide WHO
trains and WHO scores, and nothing else.

### A spec is not a plan

`spatial_cv()` returns a description; `resolve_resampling()` turns it into a
`fold_plan` against real points. Keeping them apart is what lets
`block_size = "auto"` mean "measure it when you see the data" instead of
"guess now".

### The one rule

**Every default is either obviously right or computed from the data -- never a
number someone once measured on another dataset.** That is not a style
preference; it is the lesson of the block size that was measured on 41,385
points and carried into a 10% draw.

So `block_size = "auto"` measures, `buffer = "auto"` is
`max(window) * cell_size` (the exact SQUARE separation distance -- half of it,
or a radius, leaves the diagonal sharing pixels while the report shows zero),
both print what they chose, and `"auto"` without a resolution **refuses** rather
than inventing one.

The buffer errs WIDE when the grid does not exist yet: a buffer larger than
necessary drops a few more training points; one too small reports a clean zero
for leakage that is happening.

### Not migrated tonight, on purpose

`03_run_tuning.R` still uses the low-level calls. It runs next, on code that
has been reviewed, and rewriting the trained path hours before the run is the
move that caused five restarts. `examples/quickstart.R` shows the new API;
the numbered scripts migrate after the dev run proves the pipeline.

---

## 4. kNNDM: the feasibility note was too pessimistic

The estimate in Phase 4.3 assumed O(n^2) geodesic distance matrices. Linnenbrink
et al. (2024) report kNNDM fold assignment **plus model training** on 4,000
strongly clustered points dropping from 4.8 days (NNDM LOO) to **1.2 minutes**
(kNNDM). At 3,766 dev points this is not an affordability question at all.

What stands from that note is the part that was never about cost: the points
are global lon/lat, and **a degree of longitude is 111 km at the equator and 0
at the pole**. Project to equal-area first. That is correct regardless.

---

## Sources

- Wang et al. (2025). The problematic case of data leakage: a case for
  leave-profile-out cross-validation in 3-dimensional digital soil mapping.
  *Geoderma*. https://www.sciencedirect.com/science/article/pii/S0016706125000618
- Meyer, H. & Pebesma, E. (2021). Predicting into unknown space? Estimating the
  area of applicability of spatial prediction models. *Methods in Ecology and
  Evolution* 12:1620-1633. https://arxiv.org/pdf/2005.07939
- CAST: Area of applicability tutorial.
  https://hannameyer.github.io/CAST/articles/cast04-AOA-tutorial.html
- Piikki et al. (2021). Perspectives on validation in digital soil mapping of
  continuous attributes -- a review. *Soil Use and Management*.
  https://bsssjournals.onlinelibrary.wiley.com/doi/full/10.1111/sum.12694

---

## The blame report said "alone" while reporting the not-alone number (2026-09-15)

Stage 02 finished the dev extraction and ended on this:

    WARNING: 'aboveground_biomass_carbon' alone invalidated 1.009% of points.
    ... Check its coverage over land BEFORE committing to a full prediction run.

The table printed directly above it said `n_sole_cause = 0` for every channel.

### The two situations that look identical and mean the opposite

`pct_invalidated` counts every point where a channel was non-finite --
**including points where all 181 were**. So:

| pattern | what it means | actionable? |
|---|---|---|
| one channel with a high **sole** cause | sparse NA over land; the full-window rule turns each NA pixel into a hole of up to 15x15 in the MAP | yes -- check that channel |
| every channel tied on `pct_invalidated`, zero sole causes | those points sit where the WHOLE stack is nodata: coastline, inland water, the raster's own edge | no -- dropping a predictor recovers none of them |

This run is the second case: 38 points, all 181 channels, **zero** sole causes.
The message named a channel anyway -- and it named the one that sorts first
alphabetically, because all 181 tie.

### Fixed in three places

- **02** now branches on `pct_sole_cause`. With a real sole cause it warns as
  before; with none it prints a NOTE saying what the losses actually are.
- **The 99** thresholded `pct_invalidated` at warn > 1, so it would have WARNed
  at 1.01% about nothing. It now thresholds the sole-cause column
  (warn > 0.5, fail > 2) and reports the nodata case as information --
  a WARN nobody can act on is one people learn to scroll past.
- **The snapshot** recorded `02_pior_canal` as
  `blame$predictor[which.max(blame$pct_invalidated)]`. Under a tie that is
  whichever predictor sorts first: rename one and the snapshot reports a change
  that did not happen. It now records the sole-cause channel, or `"(none)"`.

The snapshot keys changed name (`02_pior_canal` -> `02_worst_sole_cause_channel`
and friends, also out of Portuguese), so the first comparison after this will
list them as removed and new. That is correct: the quantity changed.

### What the extraction itself reported

3,766 -> 3,728 points, **1.01% removed**, well under the 2% warn line. Reading
plan: 42 of 64 chunks held a point, 66.8% of the raster's rows, 7,602 strip
reads, **4.09 hours**. The store is 1.7 GB across three windows, all three
verified by size.

And a number worth recording: the rasters are at **0.00224579811 degrees**, not
the 1/480 = 0.00208333 assumed in conversation. That is 250 m at the equator
(250/111320), so the data was right and the mental arithmetic was not. It moves
the auto buffer from 0.03125 to **0.0336870** degrees. Nothing in the code
changed -- `cell_size` is read from the raster, which is why this was a comment
error rather than a defect.

---

## `spatial_cv(k = 5)` was returning a holdout (2026-09-15)

The worst defect of the whole rebuild, found by a test written hours earlier,
and it was mine from last night.

    Error in switch(spec$kind, spatial = { : EXPR must be a length 1 vector

### The mechanism

```r
.resample_spec <- function(kind, ...) {
  structure(c(list(kind = kind), list(...)), class = "resample_spec")
}
```

**R partially matches a named argument against any formal declared BEFORE
`...`.** `k` is a prefix of `kind`. So:

```r
spatial_cv(k = 5) -> .resample_spec("spatial", k = 5L, block_size = ..., ...)
                     k = 5L   partial-matches   kind
                     "spatial" has no formal left -> falls into ...
```

The spec came out with `kind = 5L` and the word `"spatial"` buried as an
unnamed element of the list.

### Why it did not error

`switch()` on a **numeric** EXPR ignores the alternative names and returns the
nth one. `switch(5L, spatial=, random=, holdout=, region=, stop(...))` returns
the fifth -- and `switch(3L, ...)` the third, which is `holdout`.

So `spatial_cv(k = 3)` produced a **valid one-fold holdout plan**, with the
right classes, the right row counts, and a `print()` that looked plausible.
Nothing anywhere complained. That is the exact shape of failure this project
keeps paying for: the wrong answer, well-formed, silent.

`region_cv()` is the only constructor that crashed, because there `k` defaults
to NULL and `switch(NULL)` cannot pretend.

### The fix

A formal declared **after** `...` can only be matched exactly -- that is the
language rule that makes this impossible:

```r
.resample_spec <- function(..., .kind) {
  stopifnot(is.character(.kind), length(.kind) == 1L)
  structure(c(list(kind = .kind), list(...)), class = "resample_spec")
}
```

The leading dot is belt and braces: no field a constructor forwards will ever
be called `.kind`. And `resolve_resampling()` now refuses a `kind` that is not
a length-1 character instead of letting `switch()` select by position.

### The assertion that was missing

`identical(spatial_cv()$kind, "spatial")` -- one line, for each of the four
constructors. The file tested what the spec *carried* (`block_size`, `buffer`,
`k`) and never that it knew *what it was*.

### Swept for the same trap

Six other functions in `R/` declare a formal before `...`. All six are benign:
five are the S3 `print(x, ...)` signature, and `safe_read_csv2(path, ...)`
forwards to `read_csv2`, which has no argument that is a prefix of `path`.
`.resample_spec` was the only one where `...` carried field names the caller
chooses freely, which is what made it dangerous.

---

## The example pipeline runs on the front end (2026-09-15)

Cassio: *"parece simples tudo funcionando, migra pra API nova que resolve"* --
and accepted a re-run of 01 if needed.

### 01 and 02 did not need re-running, and could not use the API anyway

Two separate facts, both checked rather than assumed:

**Their outputs are still valid.** The only change to `01` after it wrote them
is inside a `message()`; `git log -p` over the write lines returns nothing.
`02` started at 22:31, after the manifest lock was committed at 21:48, and the
only later change was comments -- which the running process had already parsed.

**There is nothing in the API for them.** `dsm_load`/`dsm_train`/`spatial_cv`
are about MODELLING: opening a patch store, deciding folds, fitting. Stage 01
reads a GPKG and writes CSVs; stage 02 reads rasters and writes tensors.
Neither touches that surface.

### What each script could actually take

| script | what the front end reaches |
|---|---|
| 03 | `load_all` + `dsm_load` + `spatial_cv` + `dsm_train` -- full |
| 03b | the same, three times, one per baseline family |
| 04 | `load_all` + `dsm_load`; the seed loop stays |
| 05, 07, 99 | `load_all` only |

**Why 04 keeps its own seed loop.** `dsm_train()` would express it -- one refit
fold, the selected configs, ten seeds -- but it writes a run directory laid out
for TUNING, while stage 05 reads a layout laid out for a FITTED MODEL: the
weights and the scaling together under `<run>/<config_id>/`. Moving both at
once, before either has run, is the move that costs this project its restarts.
The loop calls the same `train_one_cnn()` `dsm_train()` would.

### The test that had to come first

`dsm_load()` and `dsm_train()` had never executed. `test_api.R` covers the
specs and the `"auto"` arguments against a hand-built `dsm_data`; nothing there
opened a store or fitted anything.

`tests/test_api_run.R` (slow) builds a store on disk with the spec fields stage
02 records, loads it through `dsm_load()`, trains a two-epoch CNN through
`dsm_train()`, and asserts:

- the lock fires **through the front door** -- a wrong predictor set or target
  is refused by `dsm_load()`, not minutes later
- `cell_size` is recovered from the manifest when no raster table is given
- `buffer = "auto"` uses the store's own resolution
- the patch path and the table path produce the **same comparison shape**,
  which is the whole point of the registry
- the plan on disk is the plan that was given
- a run resumes through the front door without retraining

A wrapper is exactly the kind of code that looks obviously right and passes the
wrong argument. Writing this before migrating was the cheap half of the job.

### What the migration removed

The six-step preamble every script repeated: open the store, read two CSVs,
align the points, open a raster for its resolution, apply the lock. **Three of
this project's lost runs came from that stretch** -- a scaling read from the
wrong file, a `data_dir` one directory off, a block size carried from another
dataset. It is now one call that cannot be half-done.

---

## align_points_to_meta() refused an integer id (2026-09-15)

The new front-end smoke test failed on its first run:

    identical(out$sample_id, meta$sample_id) is not TRUE

### What it is

`align_points_to_meta()` reorders the point table to match the patch store.
Stage 02 drops points whose window was not fully valid, so the point table
always has MORE rows and a different order -- and **every fold index in this
framework is a position in the STORE**. Get the alignment wrong and the model
trains on one point's covariates and is scored against another's target,
silently, with a plausible CCC.

### The defect

The final check was `stopifnot(identical(out$sample_id, meta$sample_id))`, and
`identical()` compares storage TYPE as well as value.

In this pipeline both sides arrive from `read_csv2()` as doubles, so it always
passed. A point table built in R carries **integer** ids against a store read
from CSV, and the check then fails on `1L` vs `1` -- with the message
`identical(...) is not TRUE`, which names nothing and suggests nothing.

That is a defect for exactly the audience this is being built for: a package
user assembling their own point table in R rather than reading one of our CSVs.

### The fix

An id is a label. Compared numerically when both sides are numbers, as text
otherwise -- and never `as.character()` on numbers, because
`as.character(1e5)` is `"1e+05"` while `as.character(100000L)` is `"100000"`,
and a framework that breaks above 99,999 points is worse than one that breaks
loudly. On a real mismatch it now names the position and both values.

### It had no test at all

That is why this survived. `align_points_to_meta` appeared in no test file --
only inside pipeline scripts, where both sides happened to be doubles.

`test_patch_store_io.R` now covers it: reordering, subsetting, integer vs
double, character ids, and a store point with no matching row. Six assertions
for a function every fold index depends on.

---

## A column named `model` shadowed the model (2026-09-15)

    Error in model$count_params : $ operator is invalid for atomic vectors

Third defect in an hour, third one found by running code that had never run.

### The mechanism

`tibble()` evaluates its arguments **in order**, and puts each finished column
into the data mask for the ones that follow.

```r
tibble::tibble(
  unit_id = unit_id, config_id = cfg$config_id,
  model = model$name,                 # creates a COLUMN called `model`
  ...
  n_params = if (!is.null(model$count_params)) ...   # resolves to the COLUMN
)
```

`model = model$name` works -- the right-hand side is evaluated before the
column exists. Every argument **after** it sees a character vector where the
model_spec used to be.

It would have fired on the first RF unit of `03b`.

### The fix, and what it is not

Not renaming the column: `model` is the right name for it. The rule is to stop
reading the spec once a column could be called that -- so `model_name` and
`n_params_val` are computed before the tibble, where `model` can only mean the
spec.

This is the second instance of the same shape today: `print_block_choice()` had
`chosen = ifelse(block_size == as.numeric(chosen), ...)` inside a `mutate()`
that also creates `chosen`.

### Swept for it

A static scan for "column name later used with `$` in the same call" returned
four more. Three are false positives -- `m$parameters$parameter` inside a
lambda, and two matches that crossed comment blocks outside any tibble. The
fourth, in `05`, is `device = device$type` as the LAST argument: safe today,
and safe only while it stays last, so it is now read before the tibble.

The durable guard is not the scan. It is `test_api_run.R`, which exercises the
table path end to end -- and which is what caught this.

### Why this run of defects is not bad news

Three real framework defects in an hour, and **none of them would have appeared
as an error in the pipeline**: the `spatial_cv` one returned a valid holdout,
the `align_points_to_meta` one only bites a point table built in R, and this
one lives in a baseline the CNN path never touches. They appeared because the
new code was exercised on purpose, before it was trusted.

---

## Pendente

| etapa | o quê |
|---|---|
| 5 | `one_se()` (o piso de ruído já é medido pelo `seed_noise_floor()`) |
| 6 | predição: inverter loops → 5 sementes → FCN |
| 7 | promover predição a função; `_scratch_1km_test/` some |
| 8 | paralelismo sobre configs (medir antes) |
| 9 | virar pacote |
| 10 | importância de variáveis |

---

## rf_grid drew the same config four times; resume matched on the label

**What broke.** The 03b board showed `rf_001`, `rf_003` and `rf_004` with CCC
identical to four decimals. They were identical *models*: `rf_grid()` sampled
`mtry_frac` and `min_node_size` with replacement and never de-duplicated, and
with `tune_length = 4` it drew `mtry_frac = 0.1` every time. Each RF family
therefore tested two distinct forests, not four, and spent 75% of its budget
re-measuring one of them.

`make_tune_grid()` de-duplicates explicitly and says why in a comment. The
baseline grids were written later and did not inherit it.

**Why it mattered beyond the wasted budget.** Every draw landed on the
*smallest* `mtry` in the pool, 0.1p, against a regression default of p/3. The
baseline the CNN was measured against was not merely narrow, it was weak -- in
the direction that confirms the project's hypothesis. A baseline that is
accidentally handicapped does not fail loudly; it agrees with you.

**Fix.** `rf_grid()` no longer draws. A forest has two parameters with a handful
of sensible values each, so the space is *covered*, ordered outward from the
defaults: `tune_length = 1` gives the textbook forest (`mtry = p/3`,
`min.node.size = 5`) and each increment adds the next most informative setting.
Reproducible, cheaper, and the seed argument is now vestigial. `mlp_grid()`
stays a random draw -- four interacting axes are what random search is for --
but de-duplicates.

*Discarded:* keeping the draw and raising `max_tries` the way `make_tune_grid()`
does. It works, but it buys nothing here: the space has 28 points, so drawing
from it is strictly worse than enumerating it, and a random baseline is harder
to defend in a paper than a default one.

**The second defect, found while fixing the first.** Renumbering the configs
changed what `rf_001` *means*. Both runners resume by `unit_id`, which is built
from `config_id` -- so resuming would have matched the old rows by name, skipped
the work, and reported results for hyperparameters that were never fitted. No
error, plausible CCC.

`.resumable_units()` (in `R/utils.R`) now matches on the hyperparameters and
lets the label follow: a cached unit is reusable only if the config it recorded
still equals the config the grid asks for under that name. Anything else is
refitted, with a message saying how many and why. It refits rather than stops,
because changing a grid is normal and stopping would punish it.

The same hazard was latent in the CNN: `make_tune_grid()` draws sequentially, so
raising `tune_length` preserves the earlier configs and resume works -- but only
while the parameter space is unchanged. Adding one value to one axis shifts
every subsequent draw, silently. That is now checked rather than assumed, which
matters immediately: the next step is exactly a larger CNN grid resuming the
existing 27 units.

*Near-miss worth recording:* the first version of the guard compared the grid's
`window_sizes` list-column `c(5L, 7L)` against the comparison row's flattened
`"5x7"`. Every CNN unit ever written would have been declared stale and
retrained -- hours of GPU time, looking exactly like a successful resume. A
guard against a silent wrong answer had become a guarantee of expensive
pointless work. Both sides are now reduced to the same tokens, and there is a
test for that specific representation mismatch.

**Statistics.** 03b compared two means against the seed noise floor. The floor
is the spread of one family across seeds -- not the standard error of a
difference -- so it can hide a real effect as easily as invent one. Every family
runs on the *same* folds with the *same* seeds, so the units are matched pairs
and most of the between-unit spread is fold difficulty that both families feel
together. `paired_family_test()` subtracts it out and reports a difference with
a confidence interval, on both CCC and MAE, because the first run ranked the
families in opposite orders on the two metrics. The interval is the point: with
9 pairs, "not significant" cannot distinguish "there is nothing" from "we could
not see it", and the interval says how large an effect is still permitted.

---

## The buffer protected the validation set and left the test set exposed

`apply_buffer()` measured the distance from every training point to the
**validation** points and dropped what fell inside the buffer. It never looked
at the test set. So a fold plan could print

```
buffer: 319 training point(s) dropped (5.1% per fold on average)
```

and the pipeline could report zero leakage — truthfully, about validation —
while every training point beside a test block kept its place. At 250 m a 15×15
patch spans 1.7 km, so those patches overlapped test patches pixel for pixel.

The asymmetry is what makes this worse than a plain omission: **the set that was
protected is the one used to choose, and the set that was not protected is the
one whose number gets published.** Every test metric measured before this commit
is optimistic by an unknown amount, and nothing in the output said so.

It is the same shape as §A0 of the September review (the buffer measuring a
circle where the patch is a square): a guarantee that is written down, believed,
and not delivered. Both were found the same way — by asking what the code
actually computes rather than what its name claims.

**Fix.** `apply_buffer(protect = c("validation", "test"))`. Training points are
now dropped for being near validation *or* near test, and the two causes are
reported separately, because one number cannot say which promise it paid for.

Validation points near the test set are dropped too. That link is weaker — the
model never fits validation rows, it only decides *when to stop* on them — but a
stopping epoch chosen on rows that overlap the test set is a small read of the
test set, and closing it is cheap: the test is carved as whole blocks, so the
affected points are a thin rim.

*Discarded:* buffering only train-against-test and leaving validation alone. It
is defensible, and it is the kind of "defensible" that is impossible to explain
in a methods section without sounding like a caveat. The cost of being strict
here is a few dozen points.

The regression test is built so the old behaviour **cannot** pass: the fixture
places training points next to the test set and far from validation, so a
validation-only buffer drops nothing, and `protect = "validation"` is asserted
to differ from the default.

**Consequence for results already recorded.** `docs/reference_performance.md`
reports test metrics from runs made under the old buffer. They are not wrong as
records of what happened; they answer a weaker question than their names imply,
and are marked accordingly.

---

## Spatial occlusion: asking the trained network whether it uses the neighbourhood

Item 9 of the September review (§D3), and the one it called "the question the
whole project needs to answer, and it has not been asked yet."

Stage 03b answers it from the outside, by racing the CNN against a forest fed
the same neighbourhood as per-channel means. `spatial_occlusion()` answers it
from the inside: hide part of the patch, re-predict, and see what the loss of
that part costs. Per Chebyshev ring, so the report also says *how far out* the
neighbourhood still matters — which is what should set the window size of the
next run. Inference only; nothing is retrained.

**Permutation, not zeros.** After scaling, zero is the training mean, so zeroing
reads as "the average landscape" — but a patch whose rim is exactly the mean
everywhere is a landscape that does not exist. A drop measured that way
confounds "this region mattered" with "this input is off-distribution", and the
second effect grows with the area hidden, which is precisely the comparison
being made. So the occluded region is taken whole from another sample: each
channel keeps its marginal and its texture, and only the association with *this*
target is destroyed. `method = "zero"` is kept because the two disagreeing is
itself a finding — a large zero-effect with a small permutation-effect says the
network is sensitive to the input being unusual, not to the content.

### Three defects found writing it, all mine, in order of seriousness

**1. The cache holds torch tensors, not base R arrays.** The function was
written against arrays and would have failed on real data. The test fixture was
written against arrays too, so both were wrong the same way and the test could
not see it. *A fixture that does not match the real representation tests the
fixture.* `occlude_patch_array()` now refuses a base array rather than accepting
one silently.

**2. Row-major versus column-major.** The natural fix — flatten `[n, c, w, w]`
to `[n, c, w*w]` and index with `as.vector(mask)` — is wrong: torch is
row-major, R arrays are column-major, so the two walk the pixels in different
orders. It would have worked anyway here, because `patch_ring_index()` is
symmetric (`outer(d, d, pmax)`), so the bug would have hidden behind exactly the
masks this file uses and appeared on the first asymmetric one. Replaced by
`torch_where()` broadcasting a `[w, w]` mask, where there is no linear index to
get backwards.

**3. `predict_loader()` floors predictions at zero.** Correct for a stock, wrong
for anything that can go negative. Left implicit, the diagnostic would floor
half the predictions of such a target and report a collapse the model never had
— and the collapse would look like a finding. `clamp` is now an argument of
`spatial_occlusion()`, with tests for both settings.

*Test-design note:* the end-to-end tests use models whose answer is known by
construction — one that reads only the centre pixel, one that reads only ring 1.
Each must report the mirror image of the other. A diagnostic that answers this
question wrongly is worse than not having it, because the answer is the sort
nobody double-checks: it agrees with whatever the reader already suspected.

---

## Calibrated uncertainty: split conformal + PICP

Item 10 of the September review (§D2), which called it the best return on effort
in the document.

**The defect it replaces.** Stage 04 reports the median and spread of a seed
ensemble, and `design_decisions.md` §11 already recorded that the spread is not a
prediction interval: it measures how much the answer moves when the
initialisation moves — a property of the optimiser, not of the soil. It says
nothing about irreducible noise, about bias, or about anything the ensemble
agrees on while being wrong together. This project's own numbers show the scale:
the seed spread is 0.038 CCC while the MAE is ~17 t/ha against a median stock of
29.3. A map that understates its uncertainty is worse than no map, because
somebody acts on it.

**What was implemented.** `conformal_calibrate()` / `conformal_interval()` /
`picp()` / `picp_report()` / `conformal_cv()`.

Three decisions worth recording:

1. **The `(n+1)` correction is not a detail.** The quantile is the
   `ceil((n+1)(1-alpha))`-th residual, not the sample quantile — the 91st of 100
   rather than the 90th. The plain quantile makes the guarantee false by about
   `1/n`, in the optimistic direction. Below `ceiling(1/alpha) - 1` calibration
   points (9 for 90%, 19 for 95%) no finite interval carries the guarantee, and
   the function returns `Inf` with a warning rather than a comfortable lie.
2. **Normalised intervals give the seed spread a job it can actually do.**
   Dividing residuals by a per-point difficulty score before taking the quantile
   makes widths vary by location while keeping the guarantee. The ensemble
   spread is useless as an interval and is a perfectly good difficulty score —
   calibrated instead of trusted. Measured on a heteroscedastic fixture, the
   easy-vs-hard coverage gap falls from 0.206 to 0.005.
3. **The guarantee is marginal, not conditional.** "90% overall" is compatible
   with 99% over the easy half and 60% over the hard half, and the hard half is
   where anyone needs an interval. `picp_report()` therefore breaks coverage
   down by group, sorts the worst first, and points at the area of
   applicability when a group falls far below — that is where exchangeability,
   the theorem's only assumption, stops holding.

*Test design:* coverage is verified by simulation over 300 repetitions, against
`k/(n+1) = 0.9020` rather than against a remembered 0.9, and repeated with a
skewed exponential error because distribution-freeness is the selling point. An
uncertainty map is the one output nobody can check by eye: one that promises 90%
and delivers 61% looks exactly like one that delivers 90%.

---

## kNNDM: folds shaped by where the map will be predicted

Item 11 of the September review (§A2).

**The weakness it addresses.** `spatial_folds()` works and is understandable
from memory, but the block size and the buffer width are *choices*, and nothing
in the data says whether they were the right ones — this project picked a block
size by measuring fold balance, which is a criterion of convenience, not of
validity. kNNDM (Linnenbrink et al. 2024, GMD 17:5897–5912) starts from a better
premise: the right validation depends on where you will predict. It shapes the
folds so the distance from a validation point to its nearest training point is
distributed like the distance from a *prediction pixel* to its nearest training
point, minimising the Wasserstein statistic W between the two.

The consequence that makes it worth a dependency: when the samples are well
spread over the prediction area, kNNDM converges by itself to ordinary random
k-fold. It does not impose separation that prediction will not face — which is
the correct criticism of blind blocking, and something a buffer cannot decide.

**Four decisions.**

1. **The algorithm is not re-derived.** `knndm_folds()` calls `CAST::knndm()`. A
   published CV method re-coded locally is a method that quietly differs from
   the one being cited, and a test written here would not catch it, because the
   test would share the misunderstanding. What is tested is everything *around*
   the call, which is where this framework can be wrong on its own.
2. **`predpoints` has no default.** kNNDM without a prediction area is an
   expensive random split wearing the name of a spatial method — the single most
   damaging convenience this file could have offered. `prediction_sample()`
   produces one from the prediction raster.
3. **The projection is not a detail.** Distances have to mean something, and a
   degree of longitude is 100.1 km at the equator against 64.8 km at 60°N
   (checked against the Mollweide formulas, ratio 0.647). Coordinates are
   projected to an equal-area projection before anything is measured, and the
   projection used is recorded in the plan. Beyond meaning, it is also cost:
   planar nearest-neighbour search is O(n log n) with a kd-tree and O(n²) in
   time *and memory* on a sphere — 31,000 points is ~7.7 GB for one matrix.
4. **There is no `buffer` argument, and that is the method rather than an
   omission.** A buffer stops a validation point sitting beside a training
   point; kNNDM's premise is that whether that is a problem depends on
   prediction, and this map is predicted wall to wall, so prediction pixels *do*
   sit beside training points. Forcing them apart would measure a scenario that
   never happens. What a buffer also caught — two profiles in the same raster
   cell, identical input and different target — kNNDM does not address, and that
   stays where it belongs, in `spatial_overlap_report()`.

*Argument validation runs before the dependency check*, so a person with a typo
gets the useful message rather than an install instruction, and the argument
contract stays testable on a machine without CAST.

W is recorded in the plan's params and printed. It is in coordinate units, so it
compares plans over the same points and means nothing across datasets.

---

## Resuming onto a different split

`.resumable_units()` proves a cached unit was fitted on the hyperparameters its
name claims. It says nothing about the **data** those hyperparameters were
fitted to — and the fold plan is data.

It surfaced the moment the buffer was fixed. Once `apply_buffer()` also
protected the test set, every fold lost a rim of training points. The cached
units were still `cfg_002` with the same learning rate and window, so the
hyperparameter check passed them, while they had been trained on a training set
that no longer exists. Resuming would have produced a comparison table whose
rows were fitted on different data and ranked against each other, with nothing
on screen saying so.

`check_plan_unchanged()` compares fold **membership** — which row indices are in
train, validation and test — rather than the plan object: params differ for
irrelevant reasons (a new field, a rounded buffer) while the split is identical,
and the split is what training consumed.

It **refuses** rather than silently discarding the cache. A run directory is the
record of an experiment; quietly replacing half of it with units from a
different experiment is worse than stopping. The caller picks a new `run_id` or
deletes the old one deliberately. The message prints both plans' sizes, because
"the plan changed" sends someone to diff two RDS files.

**It immediately found a latent bug in this project's own test.**
`test_resample_run.R` had:

```r
res1  <- ... plan = holdout(meta, validation_frac = 0.2, test_frac = 0.2, seed = 3)
res1b <- ... plan = holdout(meta)        # 0.15 / 0.15 / seed 42
             run_id = "smoke_holdout"    # the same directory
```

The test named "resume skips finished units" was resuming onto a different
split, and passed, because skipping is keyed on `unit_id`. The defect the guard
exists to catch was hiding inside the test meant to protect against it.

**The pattern, third time today.** In `spatial_occlusion()` the function and its
fixture were both written against base R arrays when the cache holds torch
tensors. Here the implementation and the test shared the assumption that resume
only depends on `unit_id`. In both cases the test passed by sharing the mistake.
What broke the pattern was a check from outside the pair — torch's own error the
first time, this new guard the second.

**Consequence for the runs ahead:** the 27 cached CNN units cannot be reused,
because the buffer fix changed the folds. The larger grid starts from zero
(~7 h at `tune_length = 24` rather than ~6 h resumed). That is the correct price
of having fixed the buffer; the cheaper alternative was a meaningless result.


## Three things found by reading the smearing wiring back, before running it

The smearing estimator was wired into stages 04 and 05 and the suite passed
21/21. Re-reading the diff before spending ~45 min on a re-run turned up three
defects, none of which any test would have caught, because all three concern
*which artefacts a run produces* rather than what any function computes.

### 1. The new block inherited the wrong gate

`smearing_from_run()` and `safe_write_csv2(ens, ...)` both sat inside

```r
if (nrow(cal_rows) >= 9L && nrow(chk_rows) > 0L) {   # the CONFORMAL condition
```

The smearing factor is calibrated on the **tuning** run's out-of-fold residuals
— a different run, a different set of points. This run's conformal calibration
count says nothing about whether it can be computed. `ensemble_predictions.csv`
is not a conformal artefact either; it is the table every later stage reads.

The gate mattered because `evaluate_test = FALSE` is now a supported mode:
`chk_rows` is empty, the condition is FALSE, and stage 05 would have found no
`smearing.rds`, written the median map alone, and printed a polite note calling
it a deliberate choice. Both blocks moved up one nesting level.

*Discarded alternative:* keeping the gate and adding `|| !evaluate_test`. That
encodes the same confusion in a longer expression — the two things were never
related.

### 2. `conditional_mean_ton_ha` was one word from meaning the opposite

The band name shipped as `conditional_mean_ton_ha`, next to an existing
`ensemble_mean_ton_ha`:

| band | what it is |
|---|---|
| `ensemble_mean_ton_ha` | mean **across the seeds** of `expm1(pred)` — still a conditional median of the stock, still low by 24% |
| `smeared_mean_ton_ha` | the conditional **mean** of the stock — the only band that may be summed |

Someone reaching for "the mean band" to total a region would have picked the
wrong one and got a plausible number back. Renamed to `smeared_mean_ton_ha`
(band `soc_smeared_mean_ton_ha`): *smeared* is the word that discriminates, and
it is the word the correction is named after.

### 3. Two checks that counted to nine

`_b4_shard_merge_check.R` held nine band names as a literal, beside
`stopifnot(nrow(ref_sum) == 9L)`. A tenth band gives that pairing two outcomes:
stop with an arithmetic complaint that names no band, or relax the count and
never check the new band at all — which is how a band reaches a map having
passed nothing.

The 1×1 summary stage 05 writes already names every band and every file it
produced, so B4 now reads `layers` and `merged` from it. The script checks the
run in front of it rather than the run it was written against.

The **snapshot** step in the same file counted to nine too, and its failure mode
was the worse one:

```r
if (length(ref_src) == 9L) { ...snapshot... } else { message("proceeding WITHOUT
  a pixel-level reference.") }
```

A tenth band turns that condition FALSE, and the pixel comparison — the only
check in B4 that looks at *values* rather than at recorded summaries — switches
itself off while every remaining check still prints PASS. The script would have
reported success having stopped doing the thing it exists for. It is now
`length(ref_src) == n_bands`, with the else branch stopping rather than
messaging, and the summary read moved above the snapshot so the count is never a
literal again.

**The common shape.** Every unit test here asserts about a *function*. These
three are properties of the *pipeline* — which files exist, what they are
called, which of them a check looks at — and the suite is structurally unable to
see them. Reading the diff back is not a substitute for tests; it is the only
thing that covers this class at all right now.


## The smearing factor was calibrated on the wrong residuals, and that was the small problem

Stage 04 ran and printed two lines that do not agree:

```
Conformal calibration set: cross-validated residuals (all folds) -- 3092 point(s)
<smearing_cal>  held-out residuals : 9276
```

9,276 = 3,092 points x 3 seeds. `R/smearing.R` asserted the factor was
calibrated on "the same out-of-fold residuals the conformal interval uses", and
`smearing_from_run()` documented itself as reading "the same source ... and for
the same reason". Both false: `cv_residuals()` collapses seeds to the per-point
median, with a comment saying exactly why (the deployed prediction *is* the
ensemble median), and `smearing_from_run()` read the raw per-seed rows.

**The project had reasoned this out once and applied it in one file.**

### The measurable consequence, and why it is not the reason for the change

`exp()` is convex, so `mean(exp(e))` grows with `var(e)`, and a single member's
residual is noisier than the ensemble's:

| calibration set | n | sd(e) | S |
|---|---|---|---|
| per-seed rows | 9,276 | 0.6698 | 1.3594 |
| ensemble residual, per point | 3,092 | 0.6507 | 1.3461 |

The gap is +0.99% and the variance difference alone predicts +1.00%, so the
mechanism is confirmed rather than inferred. On the test set it moves the bias
from +3.7% to +2.7%.

**It was still the wrong reason to change anything.** An adversarial panel
(four independent lenses, three refuters) broke the argument at its target, and
the refutation checks out against the files. Measuring S on the *deployed*
10-seed ensemble's own held-out residuals gives **1.2590** on the refit
validation fold and **1.3890** on the test set. The target is bracketed across a
10% span that straddles both candidates, so neither is defensibly "closer".

My own extrapolation to a 10-seed factor (S ~ 1.3402) was the same error in
miniature: it held the residual mean fixed and shrank only the variance. The
direct measurements show the mean moving four times further than the variance,
in inconsistent directions. It was an assumption presented as a measurement.

**So the change is for coherence, not accuracy:** the docstring was false and
`n` overstated the evidence threefold in the single line a reader uses to judge
the factor. The value moves 1%; the honesty of the reported `n` moves from
wrong to right.

### The finding that actually matters, which is eight times larger

Duan's derivation needs the residual independent of the prediction. Here it is
not. S by quintile of the out-of-fold prediction:

```
low   1.795   1.382   1.258   1.142   1.154   high
 9.3%  13.6%   17.4%   24.0%   35.7%   <- share of the observed total
```

One scalar is set largely by the low-prediction points and applied to the
high-prediction pixels that carry 60% of the carbon. The estimator that
unbiases a **sum** — the `exp(f)`-weighted mean of `exp(e)`, and the mean
surface is the one this project documents as summable — is **1.2370**, 8% below
the unweighted factor.

**Discarded alternatives, both measured rather than argued.** Calibrated
out-of-fold, applied blind to the test set:

| factor | bias | RMSE |
|---|---|---|
| 1.3461 global | **+2.7%** | 27.00 |
| 1.2370 weighted for a total | −5.8% | 26.72 |
| per-quintile | −3.9% | 26.48 |

Both refinements push the bias *further* from zero while continuing to improve
RMSE. The cause is a transfer gap neither models: the calibration models train
on ~53% of the points (CV folds) and the deployed model on ~69% (refit split),
so the calibration residuals are larger and their structure does not carry over.
That gap dominates both repairs.

The violation is therefore **reported and not corrected**: `print.smearing_cal()`
prints the quintile profile and warns above a 25% spread, naming the weighted
factor and pointing at this note. Fixing it properly needs a calibration set
generated by the deployed model, which the current design does not produce — a
real design question, logged rather than patched.

### What the panel got wrong, recorded so nobody chases it

One lens reported, with high confidence, an "adjacent validity bug": the
conformal quantile index computed at a triplicated n. It is false.
`cv_residuals()` already collapses to 3,092 points and `04_final_model.R` feeds
those to `conformal_calibrate()` — the stage-04 output prints `3092` and the
rank `2784 of 3092`, which is the correct `(n+1)` index for 3,092. Three of the
four lenses also argued the A-vs-B gap was "within SE(S), therefore noise"; the
synthesis rejected that as the wrong variance (the comparison is paired, t=5.1)
and it should not appear in any write-up. The conclusion survived; two of its
supporting arguments did not.


## B2: the two-config branch, and an `all.equal` that tolerated 26 seconds

Options 3 and 6 of `docs/b2_two_configs_decision.md`, implemented. The branch
that writes `paired_by_seed.csv` had never executed, because no selection rule
ever returns two configs -- so reaching it means naming two by hand, and naming
them means freezing them over a `selection.rds` that is this project's evidence
that nobody chose the final model after seeing the test set.

The design: copy the tuning run, delete the COPY's frozen selection, drive the
real stage 04 against the copy in a child process, assert, then delete the copy
and prove the original never moved. Three seeds rather than ten -- the branch is
a `pivot_wider` and four subtractions, identical at either count, and ten seeds
buy power in a comparison nobody may act on because it is on the test set.
25 minutes against 1-3.5 hours.

### What the review found, and why two of them were the same shape

Four adversarial lenses, then a judge that re-checked each finding against the
repo. Five survived; two made the script's headline claim untrue, which is the
worst defect available in a file whose purpose is to prove something.

**`all.equal()` on timestamps has a 26.7-second window.** The check named "its
modification time never moved" used
`isTRUE(all.equal(as.numeric(orig_mtime), as.numeric(now_mtime)))`.
`all.equal.numeric` switches to a RELATIVE comparison once `mean(abs(target))`
exceeds the tolerance; an mtime is ~1.79e9 and the default tolerance is 1.49e-8,
so the effective absolute window is **1.49e-8 x 1.79e9 = 26.7 seconds**.

That is exactly the failure its sibling cannot see: `b2_09` compares bytes, so a
rewrite with *identical content* moves only the timestamp. The two checks were
written to cover for each other and left a joint hole. Now `identical()`, with
the trap recorded -- `all.equal` is the obvious thing for the next reader to
reach for, and nothing about it looks wrong.

**The cleanup check reported "none was made" while a directory sat on disk.**
`final_dir_made` was only assigned on the happy path, after stage 04 exits 0.
But `04_final_model.R` creates its output directory at line 197, **before** its
own validations at 201-203 and long before training -- so any failure past that
line leaves a `final_<timestamp>` behind, the cleanup is skipped, and the check
short-circuits to TRUE. The leftover is not inert: stage 05 resolves
`final_run_id = "latest"` by globbing `^final_`, so a half-built directory from
a failed B2 becomes the model the next map is drawn from. The target is now
derived from disk at cleanup time by difference against a listing taken before
the run -- and nothing outside that set is touched, so a concurrent stage 04 of
Cassio's is never swept up.

**A deleted original would have killed the script instead of reporting it.** The
final `readBin()` sits outside the `tryCatch`, so the single catastrophe the
script exists to detect would have surfaced as a connection error with no
ledger, no verdict and no FAIL.

**A comment claimed a check that did not exist.** The justification for putting
the copy inside the work tree said the copy's record would carry a git commit
"which is one of the things asserted below". Nothing asserted it. B2 is the only
run in the project that exercises the `git -C` fix end to end -- a child process
writing a record into a directory inside the work tree -- so a green B2 read as
covering that fix and covered none of it. Now `b2_13`.

**One of four columns was verified while the ledger said four.** The
recomputation checked `d_ccc` only; `d_mae`, `d_rmse` and `d_mqi` rested on a
check that four column NAMES exist, which no arithmetic error can fail. Stage 04
computes them as four near-identical `paste0()` lines, which is exactly where a
copy-paste sign inversion lives. Widened to all four in the same pivot.

Two more found by re-reading afterwards: `seed` arrives as text from
`read_csv2()` and would have joined to nothing against the integer column,
failing `b2_08` for a reason that is not the defect it hunts; and the
subtraction's direction is only meaningful if `c1` is the config the script
asked for first, so the order is now tied back to the request.

### The provenance fix this uncovered

`freeze_selection()` captured `git_commit` with a bare `system2("git",
"rev-parse", ...)`, which resolves against the PROCESS's working directory. The
field was therefore NA whenever R sat outside the repo -- silently, in the one
field that exists for a third party to check the claim. Noticed only because a
test run printed `config, rule, metric, time` where the run before had printed
`..., commit 3649fd8`; nothing failed, the provenance just quietly thinned. Now
resolved with `git -C` against the record's own directory, and the message says
when it could not be resolved at all.

## 2026-09-19 — Overnight review, batch 1: the framework stops where it used to shrug

Six-lens audit over the whole repository (contracts, error paths, resume,
documentation drift, tests, ease of use), 105 findings, each re-read at the
line before anything was changed. The items below are the ones that could turn
a wrong result into one that looks right; the rest are in
`docs/status_and_roadmap.md` §3.

### A locked file is an error, not a rename

`safe_write_csv2`, `safe_save_rds` and `safe_torch_save` diverted to
`<stem>_<timestamp>.<ext>` when the target could not be removed (a Windows
handle held by Excel) and returned the new path invisibly. No caller read that
return value. So the authoritative file kept its OLD contents while a fresh one
sat beside it unread, and stage 04 would have ranked a stale
`comparison_ranked.csv` without a word. Now `.refuse_locked()` stops and names
the file. `.timestamped_path()` is gone with its two tests; the replacement
test locks the target with a directory (portable — an open handle only blocks
on Windows) and asserts that nothing was written beside it.

*Alternative considered:* keep the divert but warn. Rejected: a warning during
a 7-hour run scrolls off; the rename still leaves two files with one name.

### `%>%` is bound in the framework

`load_all.R` attaches nothing, and nine modules use the pipe, so a session
without `library(dplyr)` died at the first `%>%` — after the store had loaded,
in the README's own Quickstart. `utils.R` now binds `` `%>%` <- dplyr::`%>%` ``
once. A later `library(dplyr)` rebinds the same function.

### Three doors checked at the door

- `load_patch_store()` reads `manifest$store_complete`. Stage 02 writes FALSE
  when a window failed to save and stops — and the loader opened the store
  anyway, because nothing looked. Its "missing file" message also named
  `patches_<window>.pt`, a format this project never had.
- `dsm_load()` checks `type_table` for `predictor`, `is_dummy`,
  `is_percentage` (logical). A missing `is_dummy` died inside `case_when()`
  when the first fold cache was built, minutes in. A `raster_table` whose
  raster is missing, or without `terra`, used to fall through silently to
  the manifest's cell size; it now says which of the three it was.
- `dsm_train()` refuses a non-`dsm_data`, a non-`model_spec`, a caret-style
  `resampling = "cv"`, and any `...` name that `train_one_cnn()` does not
  take. A misspelt `patiense = 50` used to travel through two layers and fail
  at the first unit, after the plan was written and the fold cache built.

### `check_plan_unchanged()` no longer answers "fine" to a plan it cannot read

It caught the read error and returned TRUE. Now it stops: the one case in
which the check exists is the case it could not perform.

### One home for "which config did the final run deploy"

`selected_config_id(summary, label)` in `utils.R`. Three copies existed
(05, `_b1`, `_b4`), and 05b, 06, 07 and `05a_test` read
`selected_cfgs$config_id[1]` — the GRID's order, in which the runner-up can
come first when two configs were fitted. All seven sites call the helper;
`"auto"` can no longer travel on unresolved. Tested with a two-config summary
in both orders.

### A missing seed is an error; stage 04 records what it fitted

Stage 05 warned "using only 2 available" and built the map, while
`final_run_summary.rds` still listed three seeds. Stage 04 now stops when a
requested seed did not finish (the tryCatch that swallowed it stays, so the
error is printed first), writes `seeds_fitted`, and 05 refuses a checkpoint
set that differs from it.

### `setup_torch_device()` reads the machine

The default was 8 and every example script overrode it with 30 — one
workstation's number. NULL now means physical cores minus one.

### Smaller

- `06_avaliacao_grafica.R` was the eleventh alphabetical "latest" site; now
  `latest_run_dir()`.
- Raw `Sys.getenv()` reads in 05, 07 and `_b1` go through `env_chr`/`env_int`,
  so `"two"` is refused instead of becoming NA.
- `utils/install_load_pkg.R` used `require()`, which returns FALSE and let the
  banner print "completed"; now `library()` with a stop that names torch's
  `install_torch()` when that is the missing piece.
- 21 strings across `R/`, `tests/` and `examples/` had a literal line break
  where a newline escape had been — a heredoc artefact. Joined. Two Portuguese
  leftovers in English files fixed.
- The tautological test `resume_does_not_care_how_a_number_was_spelt`
  (0.0001 and 1e-4 are the same double) now respells the RECORD, where the
  value is text, and a sibling asserts that a genuinely different number is
  still caught.

Verification: `tools/r_lint.py` 0 findings, `tools/r_calls.py` 0 suspicious
arguments over 72 files. `tests/run_all.R` is the author's to run before C1.

## 2026-09-19 — Overnight review, batch 2: the checkers cannot pass by not looking

### 99_check_pipeline.R

- **A skipped stage is a row.** Four stages that had not run printed "Stage
  0X incomplete -- skipping content checks" and then the summary said "All
  clear" — true of the zero checks that ran. Each skip now adds a WARN row
  named "stage 0X outputs present", so the count and the CSV carry it.
- **The cell_size check says why it could not run.** Five conditions gated it
  (no cell_size in the manifest, no raster table, no terra, a non-numeric
  value, the first raster missing) and every one fell through silently. Each
  is now a WARN with its reason.
- **FAIL is an error.** The script ended with `warning()`, which a source()d
  run prints after the fact and exits 0 on; nobody was ever stopped by it. Now
  `stop()`, naming the report file.

### _b4_shard_merge_check.R

- **Not compared is not passed.** Without a reference snapshot the pixel
  comparison cannot run; that used to fold into `ok_pix = TRUE` and a printed
  PASS. The verdict now reads INCOMPLETE (wiring only) in that case.
- The denominators were the literal 9 of the first band count; now
  `nrow(res)`.

### 05b_merge_spatial_parts.R

- **The tile count must equal the grid the filenames declare.** A 2×2 run
  that lost a worker leaves three tiles; `terra::merge()` mosaics three tiles
  without complaint and the hole is NA that looks like ocean. Tiles from two
  grids in one directory (a 1×1 left beside a 2×2) merge into a map that is
  right where they overlap. Both refused by name, with the missing shard ids.
- A wrong mosaic geometry was a `warning()` under a written file; now an
  error that says the file must not be used.

### 05a / 05a_test / 05c

- 05a: a `prediction_config` marker whose tiles are gone no longer counts as
  "done" (the shard would have been skipped and 05b would have failed on the
  count); the eighth grid-order `selected_cfgs$config_id[1]` read →
  `selected_config_id()`.
- 05a_test passes `soc_predict_raster_dir` to its workers as 05a does; it
  used to rely on inheritance, so a 20 km test could measure the 250 m grid.
- 05c reads `max_concurrent` from the same override as 05a instead of a
  literal 3 with a "keep in step by hand" comment; an unreadable worker log
  is an error, not an empty log counted as pending work.

## 2026-09-19 — Overnight review, batch 3: tests, print methods, messages, documentation

- **`tests/test_fold_cache.R`** (22 assertions). The sentence "scaling is
  fitted on this fold's training rows" had no test: the end-to-end runs pass
  with scaling fitted on every row, because the CNN does not care where its
  z-scores came from — only the leakage argument does. Now the z-score centre
  is asserted equal to the mean of the 24 training rows and *not* equal to the
  mean of all 40; the dummy and percentage branches; the cached tensor's
  arithmetic equals the table's; the store's raw tensor survives the fold; the
  meta rows pair with the tensor rows; `store_complete = FALSE` is refused;
  `check_patch_centres()` on a consistent store, one moved value (one
  mismatch, in its channel), one NA; `clone_state_dict()` is independent and
  detached; `set_optimizer_lr()` reaches every param group.
- **Print methods** for `rf_fitted`, `mlp_fitted`, `caret_fitted`,
  `di_reference`: typing the object dumped the backend's own print.
- **Messages.** Eleven `stop()` sites in the six oldest modules leaked the
  internal call; the loader/points mismatch now names both counts and the
  helper that pairs them; `loss_fn` lists its choices; "No best_state saved"
  names the cause and the two fixes; `fit_scaling()` names the row count and
  the fix; `create_output_dirs()` names the paths. **A fold in which every
  unit failed** used to reach the ranking and die inside `dplyr::arrange()`
  over a missing `val_ccc` — minutes in, with the real cause unread in
  `error_message`. It now stops with the first error.
- **Stage 04's seed guard** checks per config, not pooled: with two configs a
  seed lost by one and kept by the other survived a pooled `unique()`.
- **Docs.** `design_decisions.md` §7's note pointed at cfg_014 (a run that no
  longer exists); §12 and `tuning_guide.md` §7 claimed the test set is written
  "for diagnostic reference" — it is not scored during tuning at all; §14 now
  says per fold. `execution_plan.md` is closed and points at
  `status_and_roadmap.md`; `test_plan.md` carries the tier status and the
  10-band note. README: nine metrics, dependencies split by layer with
  `torch::install_torch()` named, `tests/` and `tools/` beside `R/`, the run
  order, and every environment override in one table.

*Not done overnight, by decision:* the 24 example headers (self-locating
root, local installer) — a change to the scripts the author runs every
morning that nothing here can parse-check; it is item 1 of the roadmap.

## 2026-09-20 — A suíte encontrou o teste, não o framework

`tests/run_all.R`: **23/24**, 2.8 min. A única falha foi uma asserção do
`test_fold_cache.R` escrito ontem — o framework passou inteiro.

### O que falhou, e por quê

`meta_rows_pair_with_the_tensor_rows` comparava com `identical()`:

```r
identical(fpv$validation$profile_id, 25:40)
```

O meta do store volta de `read_csv2()`, que adivinha `double` para uma coluna
de números inteiros. Então o lado esquerdo é `c(25, 26, ...)` e o direito é
`25:40` — mesmos valores, tipos de armazenamento diferentes, `identical()`
FALSE. O que a asserção quer saber é se as **linhas** pareiam, e o tipo não
tem nada a ver com isso.

O detalhe que vale guardar: `align_points_to_meta()` já carrega um comentário
sobre exatamente esta armadilha, e explica que ids se comparam por valor
quando ambos os lados são números e como texto caso contrário — nunca com
`as.character()` sobre números, porque `as.character(1e5)` é "1e+05" enquanto
`as.character(100000L)` é "100000". O teste novo caiu na armadilha que o
código já documentava. Corrigido por valor, e ganhou uma asserção irmã: um
índice embaralhado tem de reordenar as linhas do meta junto.

### O que a falha destapou: 12 strings que a varredura de ontem não pegou

A limpeza de ontem procurava o padrão em que a linha de continuação **começa**
com aspas. `align_points_to_meta()` tem a outra forma: a quebra cai no meio da
frase e a continuação é texto comum. Uma varredura correta — caractere a
caractere, com estado atravessando linhas — achou mais 12 em `dataset.R`,
`occlusion.R`, `resample.R` e três scripts de exemplo.

Nenhuma delas muda o texto impresso: todas produzem exatamente o que o escape
produziria. São cicatriz de heredoc, custo de legibilidade e de `grep`.
Juntadas, e o `tools/r_skeleton.py` prova mecanicamente que os seis arquivos
têm **esqueleto de código idêntico** antes e depois — só conteúdo de string
mudou.

### A regra permanente

`tools/r_lint.py` ganhou a terceira checagem, `check_multiline_string`. Foi
escolhida pelo mesmo critério das outras duas: é um erro que já aconteceu
aqui — 33 ocorrências em dois dias, todas minhas, todas pelo mesmo mecanismo
(o Bash desta sessão transforma a sequência de escape em quebra real). A
fixture do selftest já tinha uma string de duas linhas, posta lá para enganar
o contador de delimitadores; agora ela é também o caso positivo desta regra.
Selftest PASS, repositório limpo, e a regra dispara nos arquivos de ontem.

## 2026-09-20 — C1: o preço do design é 0,19 CCC, e a grade não escolhe nada

Os dois runs (`soc_0_5cm_design_spatial`, `soc_0_5cm_design_knndm`, 8 configs ×
3 folds × 3 seeds cada) terminaram e o `_c1_design_comparison.R` rodou em
2026-09-19 05:53. Os oito checks de comparabilidade passaram: mesmos configs
casados por hiperparâmetro, mesmos seeds, mesmo test set congelado, e os
designs de fato diferentes.

### O que ele mediu, e que é sólido

O nível cai **0,19 CCC** na mediana. Todos os 8 configs caem, de −0,117 a
−0,228; o melhor sob blocos é 0,480 e o melhor sob kNNDM é 0,322. Isso
confirma, por um caminho independente, o que o B1 mostrou em distância: as
folds em bloco validam um trabalho muito mais fácil do que o mapa faz.

O SE por config **triplica** sob kNNDM — 0,042 contra 0,013 — o que também faz
sentido, já que cada fold kNNDM é uma região diferente do globo.

### O que ele reportou e que os dados não sustentam

O script imprimiu "os designs discordam por mais do que o ruído de seed
explica" e `winner_changed = TRUE` (cfg_003 → cfg_002), a partir de um rho de
Spearman de −0,43 contra um teto de 0,545.

Com 8 configs esse rho tem SE ≈ 1/√7 = 0,378 e p bicaudal ≈ 0,29; o valor
crítico a 5% é 0,738. Mas o argumento decisivo é mais simples: **0 de 28 pares
de configs estão separados por 2 SE sob kNNDM**, e 3 de 28 sob blocos. Um
design que não separa nenhum par não produziu ordem alguma, e um rho calculado
sobre essa ordem está lendo ruído — qualquer que seja o valor.

O portão que deveria ter pegado isso era `ceiling_rho < 0.3`. É um limiar sem
derivação nenhuma por trás, e um teto de 0,545 passou direto por ele.

### O que mudou no script

- **`c1_09`**, novo check obrigatório: pelo menos um design separa um par de
  configs. Contagem, não limiar.
- O portão do veredicto passou a ser a contagem de pares separados. Quando
  nenhum design separa nada, a mensagem manda ler o **nível**, não a ordem;
  quando só um separa, diz que um rho entre uma ordem e um sorteio não é uma
  afirmação sobre os designs.
- O ramo do vencedor ganhou a mesma qualificação: com a grade não separada,
  `one_se()` escolhe entre empates, e trocar de vencedor não é motivo para
  revisitar seleção nenhuma. A mensagem anterior dizia exatamente isso.
- Quando os dois designs separam, o rho ainda é comparado ao teto — mas antes
  passa por `|rho| < 2/√(n−1)`, porque um rho dentro de 2 SE de zero não
  distingue concordância de discordância.
- **`seeds_for()`**: Spearman-Brown ao contrário, do teto medido para o número
  de seeds que um ranking precisaria para se reproduzir a rho 0,80. Sai no
  console e no `c1_summary.csv`.

### O número que decide a corrida científica

| design | rho com 3 seeds | seeds para rho 0,80 |
|---|---|---|
| blocos | 0,678 | **6** |
| kNNDM | 0,545 | **11** |

**Mais seeds, não mais configs.** Subir `tune_length` de 8 para 24 a 3 seeds
não compra nada enquanto o ranking não se reproduz contra si mesmo. O custo
vai como configs × folds × seeds: 8 × 3 × 11 = 264 unidades sob kNNDM, contra
as 72 que acabaram de rodar.

*Alternativa considerada e descartada:* manter o portão em rho e apenas
afrouxar o limiar (0,5 em vez de 0,3). Rejeitada pelo mesmo motivo que fez o
0,3 falhar — o número continuaria escolhido a dedo, enquanto "a grade separa
alguma coisa?" é medível direto.

## 2026-09-21 — B6: PASS 15/15, com a interrupção executada e não pedida

A primeira tentativa (ontem) falhou pelo motivo certo: o run que devia ser
interrompido rodou inteiro porque ninguém apertou Esc dentro da janela de 55 s,
e o script **detectou isso e se recusou a seguir** em vez de reportar um resume
que não houve. Custou 6 min de CPU e não produziu nada.

A correção: a fase `worker` roda este mesmo arquivo num subprocesso, e o pai o
mata. O momento não é cronometrado — o pai lê o diretório do run e mata no
instante em que o número-alvo de unidades está no disco:

```
  0.0 min | 0 de 12    0.9 min | 2 de 12    1.6 min | 4 de 12 -> kill
```

Matar o processo é uma interrupção **mais dura** que o Esc, não mais branda: o
Esc desenrola a pilha do R e o torch pode transformá-lo num erro comum que o
runner registra como unidade falha; um processo morto não registra nada, que é
o que uma queda de energia faz.

### O que o B6 mediu

15 checks, todos PASS. Os que carregam mais peso:

- **B6-2 / B6-3**: os 4 checkpoints reaproveitados não foram tocados (mtime e
  tamanho, drift 0 s) e suas 4 linhas de comparison são byte-idênticas.
- **B6-5**: **0 checkpoints órfãos**, apesar de o processo ter morrido no meio
  de uma unidade. Nada de `.pt` truncado com nome que alguém leria.
- **B6-10**: 29 colunas de identidade idênticas ao run de controle.
- **B6-11**: os resultados ficam dentro do spread de seed do próprio controle,
  com folga de três ordens de grandeza — pior coluna `val_bias_pct` a
  **0,00224** do spread.
- **B6-13/14**: um plano de folds alterado é recusado tanto por
  `check_plan_unchanged()` quanto por `dsm_train()` antes de treinar qualquer
  coisa, e o `fold_plan.rds` em cache fica intacto.
- **B6-15**: `cfg_001` relabelado com outros hiperparâmetros tem suas 2
  unidades em cache descartadas e refeitas. O id é rótulo; a configuração é a
  identidade.

### Uma observação que não é um defeito, e uma hipótese que não foi medida

A tabela de B6-11 separa as unidades reaproveitadas das retreinadas, e as duas
se comportam de modo diferente contra o controle:

| | diferença do controle |
|---|---|
| retreinadas (processo R interativo) | **0, exato** |
| reaproveitadas (worker via `Rscript`) | 1,2e-4 em `val_ccc` |

O projeto já tinha registrado determinismo entre processos (seed 7, CCC
0,480181867591615 em três runs). Isso continua valendo entre sessões
interativas — as 8 unidades retreinadas bateram exatamente. O que apareceu
agora é que a unidade treinada num **subprocesso lançado por `Rscript`** difere
na quinta casa.

Hipótese, **não medida**: ambos pedem `setup_torch_device(n_threads = 30)`, mas
o worker herda `OMP_NUM_THREADS=30` do pai *antes de o R arrancar*, enquanto o
processo interativo chama `Sys.setenv()` depois que o `load_all.R` já carregou
o torch. O OpenMP costuma ler essa variável na primeira região paralela, de
modo que os dois podem estar rodando com pools de tamanho diferente, e a ordem
de redução em ponto flutuante muda com isso.

Se for isso, tem uma consequência prática que vale medir um dia: o
`setup_torch_device()` chamado depois do torch carregar pode não estar
entregando o número de threads que diz entregar. Não afeta nenhum resultado
deste projeto — 1,2e-4 contra um ruído de seed de 0,081 é 0,15% — mas afeta a
leitura de "training is deterministic across processes", que agora é: entre
sessões interativas, sim, exatamente; contra um subprocesso, dentro de 1e-4.

*Alternativa considerada e descartada:* manter só o caminho manual e pedir de
novo o Esc. Rejeitada porque cada tentativa custa o control run mais o run
interrompido, e a taxa de acerto depende de o usuário estar olhando para a
tela no minuto certo. O caminho manual continua disponível em
`soc_b6_interrupt = "manual"`, porque o Esc exercita um modo de falha
genuinamente diferente — o erro que o runner registra e do qual continua.

## 2026-09-21 — B3: a augmentação foi medida, e o tier B fechou

Dois braços, `augment = TRUE` e `augment = FALSE`, diferindo nisso e em mais
nada: mesmo config (cfg_003, o implantado, lido da seleção congelada), mesmo
plano de folds (lido do run 03 e passado aos dois), mesmos três seeds, pareados
em (fold, seed), 9 pares por braço. 18 unidades, ~35 min.

| | val_ccc |
|---|---|
| augment = TRUE | 0,4726 |
| augment = FALSE | 0,4426 |
| diferença pareada | **+0,0300** (SE 0,0124), IC 95% [+0,0014, +0,0587] |
| | t(8) = 2,42, p = 0,042 |
| ruído de seed (braço mais duro) | 0,0533 |

`val_mae` concorda na direção (−0,044 a favor do ON) mas não separa sozinho:
IC [−0,154, +0,066].

### O que isso substitui

`design_decisions.md` §8 afirmava que a augmentação "ajudou de forma
significativa", citando a melhora de CCC entre a rodada 1 e a rodada 2
(0,569 → 0,585–0,605). Aquela rodada também mudou janelas (3/5/7 → 3/9/15),
resolução (20 km → 250 m), a grade e o schedule de treino. A augmentação era
uma de pelo menos cinco coisas que se moveram, então o número não sustentava
afirmação nenhuma sobre ela. Agora sustenta — e a direção é a mesma que o
documento sempre alegou, o que é sorte e não método.

### As duas leituras do mesmo número, e por que o script não escolhe

O veredicto impresso foi `effect_smaller_than_the_seed_noise`: +0,0300 é menor
que os 0,0533 que este modelo move entre seeds, e pela regra permanente do
projeto isso não é evidência acionável.

Essa regra nasceu para comparar CONFIGS dentro de uma grade, onde se escolhe o
máximo de muitos e o ruído vira viés de seleção. Aqui o desenho é outro: dois
braços planejados, pareados, com o pareamento removendo fold e seed — e o SE de
0,0124 do teste pareado **já contabiliza** o ruído. As duas comparações
respondem perguntas diferentes:

- *"Um treino único com augmentação bate um treino único sem?"* — 0,0300 contra
  0,0533: o ruído domina. Não.
- *"A média de um ensemble sobe?"* — é o que o teste pareado mede, e o modelo
  implantado é um ensemble de seeds, não um treino único.

**Não mexi no veredicto.** Afrouxar o critério depois de ver o resultado é
exatamente o erro que este projeto documenta em vários lugares, e o fato de a
mudança favorecer a conclusão que o autor já esperava a torna mais suspeita,
não menos. A observação fica registrada aqui; a régua fica onde estava.

### O tamanho, em contexto

+0,030 é pequeno em termos absolutos, mas vale comparar com o que o projeto já
mediu:

| o que se mexe | quanto vale em CCC |
|---|---|
| o design de validação (C1) | **0,19** |
| a augmentação D4 (B3) | **0,030** |
| toda a busca de arquitetura (C1, melhor − pior de 8 configs, blocos) | **0,042** |
| o ruído de seed | 0,053 |

Ou seja: ligar a augmentação vale quase tanto quanto **toda a amplitude da
grade de 8 arquiteturas**. E as duas coisas juntas continuam seis vezes menores
que a escolha de como validar.

A augmentação fica ligada. Custa praticamente nada em CPU (o early stopping
parou em épocas comparáveis nos dois braços), a direção medida é positiva, e o
prior que ela codifica é fisicamente correto. O que mudou é que isso agora é
uma medição com intervalo, e não uma crença com uma citação que não a
sustentava.

### Um subproduto: determinismo confirmado na quinta casa

O braço ON reproduziu as mesmas 9 unidades do run `soc_0_5cm_20260916_232318`
com `|diferença| máxima = 1,7e-6` em val_ccc. `R/train_cnn.R` e `R/metrics.R`
mudaram depois daquele run, e mesmo assim nada se moveu. Isso confirma pelo
terceiro caminho independente o que o B6 mediu: entre sessões interativas o
treino é determinístico.

## 2026-09-21 — Os 26 cabeçalhos: o projeto para de morar numa máquina só

Item 1 do roadmap, o último que separava isto de virar pacote. Antes:

- **26 arquivos** com `project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"`
  escrito na mão. Ninguém além do autor conseguia rodar o projeto sem editar
  26 arquivos.
- **15 deles** buscavam `install_load_pkg()` de uma URL do GitHub *a cada
  execução* — uma dependência de rede para **começar**, num script que depois
  trabalha inteiramente em disco local. A cópia em `utils/install_load_pkg.R`
  é a mesma função.

Agora todos usam o snippet auto-localizável que `tests/*.R` e `R/load_all.R`
já usavam — provado em 26 arquivos antes desta passada — e o instalador vem
do disco. Zero buscas de rede no repositório.

### O que fazia disto mais que um find-and-replace

A ORDEM. A URL era sourceada **antes** de qualquer `project_root` existir, e o
`rm(list = ls())` que vinha depois apagaria um. Então o snippet toma o lugar
da URL, e a limpeza virou `rm(list = setdiff(ls(), "project_root"))` — a mesma
forma que o `_b1` já usava pelo seu próprio motivo, e que mantém a estrutura
de cada script reconhecível para quem a conhece.

Quatro casos não seguiam o molde:

- **`05a_run_parallel.R`** já sourceava `R/utils.R` na linha 5, antes de onde
  a URL estava. O snippet teve de ir para o topo absoluto.
- **`_b6_resume_check.R`** monta `b6_script` — o caminho que ele imprime nas
  próprias instruções — quarenta linhas antes de definir o root.
- **`06_avaliacao_grafica.R`** tem o root como **argumento default** de uma
  função. Um default é avaliado quando a função é *chamada*, e nesse momento o
  frame do `source()` — a única coisa que sabe onde o arquivo está — já se foi.
  Resolvido uma vez, em `.dlc_root`, durante o `source()`.
- **`_b4`** alinhava a atribuição com espaços extras.

### Dois bugs que eu introduzi, e o que os pegou

Escrevi `tools/check_headers.py` antes de confiar no resultado, porque as três
maneiras de errar aqui só apareceriam ao rodar o script: usar `project_root`
antes de defini-lo, um `rm()` entre a definição e o uso, e chamar
`install_load_pkg()` antes de sourcear o arquivo que a define. Tudo isso se
responde lendo números de linha.

Ele achou os dois:

1. **`05a`**: o snippet caiu depois de um uso que já existia na linha 5.
2. **`_b1`**: o `replace` trocou a primeira ocorrência da string
   `rm(list = ls())` no arquivo — que estava **dentro de um comentário**, na
   linha 79 — e deixou o código real, na 141, intacto. O comentário ficou
   corrompido e o `rm` continuou apagando o root.

O segundo é o erro clássico de editar código com busca textual, e é o mesmo
que o `r_skeleton.py` existe para pegar em outro contexto. A correção usa
âncora de início de linha (`^rm\\(list = ls\\(\\)\\)$`), que um comentário não
pode casar. Verifiquei os outros 12 arquivos com o mesmo padrão: só o `_b1`
tinha um comentário citando o `rm`.

`tools/check_headers.py` ficou no repositório, com as outras três ferramentas.

### O que continua absoluto, de propósito

`predictor_raster_dir` em 01, 02 e `_b4`. Isso é onde os dados **do usuário**
estão, não onde o código está, e nenhuma auto-localização pode descobrir. É
uma configuração e o lugar dela é visível no topo do script que precisa dela.

*Alternativa considerada e descartada:* um arquivo `examples/_root.R` único,
sourceado pelos demais, para não repetir 20 linhas em 26 arquivos. Rejeitada
por ser circular — para sourcear esse arquivo é preciso saber onde ele está,
que é exatamente o problema. A duplicação é o preço de não ter um pacote
ainda, e desaparece no dia em que `library(deep_learning_caret)` substituir
tudo isso.

### O primeiro uso real do `stop()` no 99 achou um check mal calibrado

A conversão dos cabeçalhos foi verificada rodando o `99_check_pipeline.R` num
console limpo: ele achou o projeto sozinho, carregou os pacotes do instalador
local e chegou ao resumo — 69 PASS, 3 WARN, 1 FAIL. E o FAIL foi um falso
positivo:

```
[FAIL] 04 | stage 04's tuning_run_id == stage 03's most recent run
           used_by_04=soc_0_5cm_20260916_232318 | mais_recente_03=soc_0_5cm_design_spatial
```

Nada está errado. O `soc_0_5cm_design_spatial` é um dos dois runs do C1, e C1,
B3 e B6 **todos** escrevem runs de tuning — nenhum deles candidato a
substituir o modelo de produção. Sob o `warning()` antigo ninguém reparava;
com o `stop()` que pus ontem, isso trava o pipeline por um não-problema.

A pergunta que o check fazia não é respondível por um script: só a pessoa sabe
se o run mais novo foi um experimento ou um stage 04 esquecido. Então ela
virou **WARN**, com a mensagem nomeando as duas leituras.

No lugar dela, como FAIL, entrou a pergunta que **é** respondível e que o check
antigo estava tentando alcançar sem conseguir: a config que o stage 04
implantou está no grid do run que ele diz ter usado? Isso pega um
`tuning_run_id` apontando para o run errado, um grid redesenhado sob outra
seed, e um `config_id` que hoje significa outra coisa — três falhas reais que
"é o mais recente?" nunca veria.

*Observação para depois:* o WARN do snapshot ("12 de 21 valores mudaram") é
ruído permanente — são as chaves renomeadas de português para inglês
(`01_n_linhas` → `01_n_rows`), que aparecem como 6 `gone` e 6 `new`. O
snapshot de referência precisa ser regravado uma vez, agora que os nomes
estabilizaram. É uma decisão do autor porque o snapshot é a linha de base
contra a qual tudo se compara.

### O snapshot: ninguém deveria precisar apagar um arquivo para consertar a linha de base

Ao explicar como limpar uma linha de base velha, apareceram dois defeitos no
mecanismo — e a explicação em si estava errada.

**O que eu disse errado:** que o snapshot de referência precisava ser regravado
à mão. Não precisa. O `write_run_snapshot()` grava um novo a cada execução, e o
de hoje (`snapshot_20260921_224657.csv`) já saiu com as chaves em inglês. A
próxima rodada compara contra ele e as 12 "mudanças" desaparecem sozinhas.

**Defeito 1: o snapshot anterior era escolhido por NOME.**
`sort(files, decreasing = TRUE)` — o mesmo erro que o `latest_run_dir()`
carregava até 2026-09-18, e aqui ele é alcançável e não teórico: o
`write_run_snapshot()` aceita um `label`, então um snapshot gravado como
`snapshot_baseline.csv` ordena acima de qualquer `20260921_224657` e vira a
referência permanente. Agora é por tempo.

**Defeito 2, o sério: um run com FAIL virava a linha de base.** O
`write_run_snapshot()` rodava incondicionalmente, antes do resumo e antes do
`stop()`. Então um run quebrado gravava a referência contra a qual o run
seguinte se compara — e o seguinte reportava "nada mudou", porque estava
comparando um estado ruim com o mesmo estado ruim. O momento em que um número
deslocado mais importa era exatamente o momento em que o mecanismo ficava cego
para ele.

Com isso corrigido, a linha de base nunca precisa ser apagada: ela avança
apenas em runs que passaram, e um run que falhou deixa a última referência boa
de pé. O script diz isso quando acontece.

## 2026-09-26 — Decisão: os seeds estabilizam a mediana; a incerteza vem do conformal

Registrada antes de qualquer código, porque é a escolha que um revisor vai
questionar primeiro.

**Continua-se treinando N seeds no modelo final.** O papel deles é o que a
teoria sustenta: a mediana de um ensemble é uma estimativa pontual de variância
menor que a de qualquer membro. O SD e o MAD entre seeds continuam gravados,
como diagnóstico do otimizador, e nunca como intervalo.

**O mapa de incerteza vem de predição conformal**, que é a matemática com
garantia: cobertura ≥ 1 − α em amostra finita, sem supor normalidade nem que o
modelo seja bom, exigindo só que os pontos de calibração sejam permutáveis com
os de predição.

### Por que o intervalo atual não basta

O `05` já grava um intervalo conformal (`pi90_lower`/`pi90_upper`), e ele tem
três limitações:

1. **Largura constante.** Todo pixel recebe o mesmo ± q em t/ha. Honesto na
   média, mudo sobre *onde* o modelo sabe menos.
2. **Calibrado no trabalho errado.** Os resíduos vêm da CV em blocos, a ~16 km
   do treino; o mapa prediz a ~824 km (B1). Sem permutabilidade não há
   garantia, e o erro provável é o otimista: intervalos estreitos demais longe
   dos dados.
3. **Em unidades nativas, sobre um modelo treinado em log1p** e com resíduos
   heterocedásticos — o fator de smearing vai de 1,79 a 1,15 entre os quintis
   da predição. Um ± q constante em t/ha é largo demais nos valores baixos e
   estreito demais nos altos, que carregam 60% do estoque.

### As três opções, da mais barata à mais cara

**(a) Conformal no espaço log.** Escore |log1p(y) − log1p(ŷ)|, intervalo
expm1(log1p(ŷ) ± q). A largura cresce com o nível predito, no mesmo espaço em
que o modelo foi treinado. Sem retreino. A garantia vale para qualquer escore
permutável — trocar o escore não a enfraquece.

**(b) (a) normalizado pelo DI.** O escore é dividido por uma escala que é função
do índice de dissimilaridade (`R/aoa.R`, Meyer & Pebesma 2021), calculado do
mesmo jeito num ponto de calibração e em cada pixel. A largura cresce onde o
modelo extrapola. Sem retreino. É o conformal normalizado de Papadopoulos
(2008) e Lei et al. (2018), com o escore de dificuldade que este projeto já tem.
O spread dos seeds **não** serve como escala aqui: os resíduos de calibração
vêm de modelos de CV, cujo spread tem outra escala que o do ensemble final.

**(c) CQR** (Romano, Patterson & Candès 2019). A rede passa a prever quantis
com perda pinball, e o conformal corrige a cobertura deles. Adaptativo à
heterocedasticidade do alvo e com garantia — o estado da arte. Exige mudar a
cabeça da rede e retreinar o tuning e o modelo final.

**Em todas, a fonte dos resíduos de calibração é um argumento**, não uma
constante do código: é ela que decide para qual trabalho o intervalo é
honesto. Resíduos de blocos dão um intervalo válido para interpolação perto
dos perfis; resíduos de kNNDM, para o trabalho que o mapa faz. Esta é a
segunda consequência da pergunta científica 1 — a primeira foi o número
reportado; esta é a largura do intervalo no mapa.

### O intervalo de hoje, medido — e o log testado antes de construí-lo

Recalculado a partir dos CSVs com a receita exata do 04 (`cv_residuals()` +
`conformal_calibrate()`); a cobertura total bate com os 87,8% que o 04
reportou.

- 3.092 resíduos da CV em blocos do `cfg_003`, mediana dos seeds por ponto,
  |obs − pred| em t/ha
- q90 = **39,6 t/ha** (q95 = 55,3): todo pixel recebe mediana ± 39,6, cortado
  em zero
- largura **79 t/ha em todo pixel**. Como a mediana do estoque é 29 t/ha, o
  limite inferior é **zero na maior parte do mapa**
- o SD entre os 10 seeds tem mediana **4,3 t/ha** no teste; o intervalo é 9×
  mais largo — a medida, no modelo implantado, do que o cabeçalho de
  `R/conformal.R` afirmava

Cobertura no teste congelado (591 pontos, ~118 por quintil, erro padrão ≈ 3
pontos percentuais), nominal 90%:

| quintil do predito | hoje (± q em t/ha) | log1p | escala ajustada |
|---|---|---|---|
| 1 (~10 t/ha) | 94% | 78% | 89% |
| 2 (~19) | 94% | 94% | 92% |
| 3 (~26) | 92% | 92% | 88% |
| 4 (~33) | 82% | 86% | 83% |
| 5 (~58) | **77%** | 97% | 88% |
| total | 88% | 90% | 88% |

- **Hoje**: 77% no quintil de maior estoque, onde está a maior parte do
  carbono — 4 erros padrão abaixo do prometido.
- **Log**: acerta o total, mas inverte o problema (78% no quintil baixo). Trata
  o erro como proporcional ao valor predito, e ele não é.
- **Escala ajustada**: |resíduo| = 8,59 + 0,258 × predito, ajustada numa metade
  dos resíduos de calibração; o q vem da outra metade, e é essa separação que
  mantém a garantia. Cobertura entre 83% e 92% em todos os quintis.

**Consequência:** a opção (a) do registro acima estava errada no detalhe. A
largura deve crescer com o nível, mas numa escala ajustada aos dados, e não na
imposta pelo log. A família escolhida continua a mesma — conformal normalizado,
sem retreino, com o DI como a segunda dimensão da escala.

**O que este teste não pode mostrar:** o teste congelado fica perto do treino
(blocos), então nenhuma destas coberturas vale a 824 km. É para isso que serve
o DI, e é por isso que a fonte dos resíduos de calibração continua sendo um
argumento.

## 2026-09-26 — Passo 1: `dsm_prepare()`, a receita, e o store que se carrega sozinho

### O que é

`R/prepare.R`: uma tabela de pontos (sf, data.frame ou arquivo espacial) e uma
pasta de rasters alinhados viram um store de patches. É o `01` e o `02` com o
dataset tirado de dentro: tudo o que lá era literal virou argumento, e todo
argumento é gravado no store.

O que se declara: `windows` (obrigatório), `percentage` (regex sobre os nomes
limpos), `dummy` (`"auto"` ou nomes), `drop`, `na_below`, `percentage_limits`,
`transform` (`"none"` ou `"log1p"`), `target_min`, `profile_id`, `subsample`,
`n_cores`.

**A receita** (`recipe.rds`) guarda transformação, tipos, regras de QC,
descartes, janelas, resolução, CRS e quantos núcleos a extração usou. O store
passa a carregar cópias das quatro tabelas de que precisa, então é
autocontido: `dsm_load(store)` não precisa de mais nada, e expõe
`$transform` — nome, direta e inversa — lido do que o store registrou.

### Decisões, com as alternativas que perderam

- **`windows` sem padrão.** 3/9/15 foram escolhidos para 250 m; a extensão de
  uma janela é janela × resolução. A regra do `api.R` é que nenhum padrão pode
  ser "um número medido em outro dataset". *Alternativa descartada:* manter
  3/9/15 como padrão, que funcionaria em silêncio e errado a 30 m ou a 1 km.
- **A declaração vence a detecção.** Um canal declarado como percentagem nunca
  é rebaixado a dummy porque os pontos só viram 0 e 1. O `01` precisava de uma
  lista à parte (`force_as_percentage`) para isso. Nos dados do SOC nenhum
  canal é as duas coisas, então a tabela sai igual.
- **Uma banda por núcleo, e o resultado não depende de quantos.** O `02` fazia
  bloco-fora, banda-dentro. Aqui é banda-fora: cada banda é um arquivo, cada
  ponto está num único bloco de linhas, então a banda de um worker não depende
  de nenhuma outra, e o pai remonta por índice. O teste prova `n_cores = 2`
  idêntico a `n_cores = 1`, bit a bit. As funções dos workers ficam no nível
  de cima de propósito: uma closure definida dentro do `dsm_prepare()` seria
  serializada junto com o frame dele — que segura os arrays de patches, GB
  deles — e mandada a cada worker.
- **Um store existente é recusado**, a menos que `overwrite = TRUE`. O `02`
  pulava janelas cujo arquivo já tinha o tamanho certo, para retomar depois de
  um crash; numa função isso está errado: dois stores de mesma forma,
  construídos de pontos ou preditores diferentes, têm arquivos de tamanho
  idêntico, e o pulo manteria os patches velhos sob o manifest novo.
- **Um `n_cores`, uma conta.** `resolve_cores()` no `utils.R`: núcleos físicos
  − 1 por padrão, validado, e avisa acima dos físicos. O `setup_torch_device()`
  passou a usar a mesma conta. Antes ele contava físicos − 1 enquanto o `05`
  dividia os lógicos pelos workers — duas respostas na mesma máquina.

### Três checks do `01` que nunca podiam disparar

- **Nomes duplicados.** `janitor::make_clean_names()` aplicado ao vetor inteiro
  já desduplica (`a_b`, `a_b_2`), então o `count(predictor) > 1` depois dele
  nunca achava nada: dois rasters que viram o mesmo nome eram renomeados em
  silêncio. Agora cada nome é limpo sozinho e o conflito é recusado, nomeando
  os arquivos.
- **CRS.** `if (!is.na(terra::crs(r)) == FALSE)`: o terra devolve `""` para
  "sem CRS", não `NA`, então a condição era sempre falsa. Agora `nzchar()`.
- **`has_na` no relatório de risco.** É calculado depois que as linhas com NA
  já foram removidas pelo QC, então conta zero sempre. **Reproduzido como
  está**, para a verificação contra o store atual ser limpa; a correção vai num
  commit só dela.

E uma correção ao que eu disse antes: o manifest **registrava** sim
`target_transform = "log1p"`. O defeito era que ninguém lia — o `04` e o `05`
digitavam `expm1` à mão.

### Um achado de CPU

Esta máquina tem **16 núcleos físicos e 32 lógicos**. O `03`, o `03b` e o `04`
passam `n_threads = 30` ao torch — ou seja, rodam com threads quase o dobro
dos núcleos que fazem a conta. A partir de agora o `setup_torch_device()` avisa
disso em cada run. O número não foi mudado aqui: a contagem de threads pode
mexer na quinta casa, e mudar isso em silêncio quebraria a comparação com
todos os runs anteriores. Fica para o passo 2, medido.

### Como isto é verificado

- `tests/test_prepare.R` (44 asserções; 43 sem o pacote sf): uma grade 30 × 40 com seis rasters
  cujo valor em cada célula é conhecido por fórmula, e 13 pontos em que cada um
  exercita um caminho — sentinela no centro, sentinela dentro da janela, borda,
  alvo zero, alvo NA, perfil repetido, percentagem acima de 100 e abaixo de 0.
  O patch é comparado com a fórmula do raster, o centro de cada patch com o
  valor do ponto (`check_patch_centres()`), e `n_cores = 2` com `n_cores = 1`.
- `examples/soc_stock_0_5cm/_p1_prepare_check.R`: roda o `dsm_prepare()` nos
  dados reais, com as configurações lidas de volta do `target_config.csv` e do
  manifest, num diretório separado, e compara 19 artefatos com o store atual.
  O `01` e o `02` só passam a chamar a função depois que isso der 19/19.

### O P1 estourou a memória, e a causa era a extração em paralelo que eu escrevi

A suíte passou (25/25), mas o `_p1_prepare_check.R` saturou a RAM e foi
interrompido. A conta:

- a grade tem **63.721 linhas × 160.298 colunas**;
- o `02` lia, por banda e por bloco de 1.000 linhas, **a largura inteira**:
  1.014 × 160.298 células = **1,3 GB** em double, para recortar algumas
  centenas de patches de 15 × 15;
- uma banda por vez isso cabia. A primeira versão paralela rodava **15 ao
  mesmo tempo**, e com as cópias temporárias que o `qc_band_values()` faz no
  caminho o pico chegava a ~**100 GB** numa máquina de 63.

A versão paralela multiplicou por 15 um custo que o `02` pagava uma vez só, e
eu não medi antes de entregar.

### O que os arquivos são por dentro

Li o cabeçalho TIFF dos 187 rasters: todos **em faixas de uma linha, LZW,
float32**. Isso decide o que se pode economizar. Ler uma janela de 15 colunas
não poupa descompressão: tocar uma linha obriga o GDAL a descomprimir as
160.298 colunas dela. Poupa tudo o que vem depois — o R deixa de converter e
alocar 160 mil doubles por linha para aproveitar 15.

### O desenho novo

- **Ler só as colunas em volta dos pontos.** Os pontos de cada bloco de linhas
  são ordenados por coluna e agrupados; cada grupo é lido como uma janela
  própria. Uma leitura fica limitada a `chunk_nrows × read_max_cols` —
  33 MB nos padrões, contra 1,3 GB —, e a memória por worker deixa de depender
  da largura do mundo.
- **Uma descompressão por linha.** Cada banda é aberta uma vez (`readStart`) e
  lida janela a janela (`readValues`). `terra::values()` abre e fecha o arquivo
  a cada chamada, e fechar o arquivo esvazia o cache do GDAL. O cache de cada
  worker é dimensionado para caber as linhas de um bloco: 682 MB no SOC.
  Sem isso, cada worker herdaria o padrão do GDAL — 5% da RAM **cada um** — e
  15 deles reservariam três quartos da máquina para cache.
- **O número de workers respeita um orçamento.** `max_ram_gb` (padrão: 70% do
  que está livre quando a extração começa, medido pelo pacote `ps`). O plano de
  RAM é impresso antes de começar.
- **`read_gap` e `read_max_cols`** controlam como os pontos são agrupados. São
  alavancas de desempenho e **nada mais**: o teste força uma leitura por ponto
  e blocos de 3 linhas, e exige arrays idênticos aos do padrão.

### Uma incerteza registrada, não suposta

O `terra` 1.9-46 instalado exporta `gdalCache()`, mas nada no disco diz a
unidade (a documentação do terra diz MB). Uma unidade errada não quebraria nem
o resultado nem a memória, só a velocidade. Então o código lê de volta o valor
aplicado e imprime ao lado do pedido — a primeira execução responde.

*Alternativas descartadas:* (1) só limitar o número de workers pela memória,
mantendo as leituras de largura inteira — funcionaria, mas com ~6 workers e
convertendo 1,3 GB por leitura; (2) ler uma janela por ponto sem agrupar —
multiplicaria as chamadas ao GDAL no conjunto completo (41 mil pontos × 181
bandas ≈ 7,4 milhões de leituras).

### P1: PASS 19/19 — o `dsm_prepare()` constrói o mesmo store

Com as configurações que o `01` e o `02` registraram, lidas de volta do
`target_config.csv` e do manifest, num diretório separado:

- as três janelas de patches **idênticas bit a bit** (`max |diff| 0`):
  3.728 × 181 × 3 × 3, × 9 × 9, × 15 × 15;
- o manifest idêntico nos 17 campos; tipos, regras de QC, risco de canal,
  culpa do full-window rule, resumo de QC, tabela de rasters: idênticos valor
  a valor e byte a byte;
- `dsm_load(store)` sozinho carrega o store e lê a transformação de volta.

Memória sob controle: 15 workers × ~1,1 GB, cache do GDAL de 683 MB por
worker, **lido de volta igual ao pedido** — a unidade do `gdalCache()` é MB,
como a documentação dizia.

### A única diferença de texto: um erro de ida e volta pelo CSV, no caminho antigo

Três arquivos saíram com "valores idênticos, bytes diferentes". Dois têm
explicação óbvia (a coluna renomeada no cabeçalho). O terceiro, o
`patch_meta.csv`, não tem coluna renomeada, e 472 das 3.728 linhas diferiam no
**último dígito** do `target_native`: `…224473` contra `…224487`.

Comparado com o valor original, lido direto do GPKG (que é um SQLite):

| perfil | GPKG | antigo | novo |
|---|---|---|---|
| 1006165 | 85.83445839122449 | −1,4e-14 | **igual** |
| 1006213 | 108.2978911764706 | −1,4e-14 | **igual** |
| 1006227 | 46.27401995351123 | +7,1e-15 | **igual** |

O caminho antigo gravava a tabela de pontos em CSV no `01` e o `02` a relia com
o `readr`, cujo leitor **não arredonda corretamente o último dígito**: 13% dos
valores voltavam deslocados de uma unidade na última casa. O `dsm_prepare()`
mantém tudo em memória, e o valor novo é o do GPKG. É também por isso que o P1
disse "valores idênticos": ele relê os dois com o mesmo `readr`, que leva os
dois textos ao mesmo double. Os patches foram comparados em RDS, exatamente.

Consequência prática: nenhuma — o alvo entra no torch em float32, com 7
dígitos. É a regra 2 do `execution_plan.md` ("CSV é formato de apresentação")
aparecendo por um caminho que ninguém esperava.

### Onde foram os 93 minutos

A extração em paralelo levou **60,6 min**; o store inteiro, **92,8**. O perfil
do progresso diz onde está o custo: as primeiras 15 bandas levaram 10,5 min, as
últimas 45 levaram menos de 4. As primeiras são contínuas (biomassa, `bio1`…
`bio19`), float de alta entropia, que o LZW descomprime devagar; as últimas são
dummies e classes, que comprimem bem. É o custo de descomprimir linhas inteiras
de 160 mil colunas — inerente a TIFFs em faixas de uma linha.

Os ~32 min fora da extração são, muito provavelmente, o `terra::extract()` dos
valores de centro nos 4.154 pontos, que roda em série e descomprime a linha de
cada ponto em cada um dos 181 arquivos. Duas acelerações possíveis, **nenhuma
feita**:

1. **Ler os centros na mesma passada paralela dos patches** — os valores de
   centro são o centro de uma janela 1 × 1, e as linhas já estão sendo
   descomprimidas. Economizaria a fase serial (~30 min aqui, horas no conjunto
   completo de 41 mil pontos). Exige repetir o P1.
2. **Converter os rasters para TIFF em blocos (tiles)** — é uma decisão sobre
   os dados de origem, não sobre o código. Num arquivo em blocos de 256 × 256,
   um patch de 15 × 15 toca de 1 a 4 blocos em vez de 15 linhas inteiras.

### Duas correções que o P1 permitiu fazer

- **`has_na` agora conta antes do descarte.** O `01` contava NA nos pontos
  depois que o QC já tinha removido toda linha com NA, então o risco nunca
  disparava. Agora conta sobre todas as linhas extraídas, depois das regras de
  QC. Isto muda o `channel_risk.csv` (de propósito); o store não muda.
- A lista de dummies detectadas, que imprimia 77 nomes numa linha, mostra 8 e
  aponta para a tabela.

### O `01` passa a chamar o `dsm_prepare()`, e o `02` deixa de existir como etapa

O `01` agora são as configurações do SOC — pastas, descartes, padrões de
percentagem, a sentinela de temperatura, as janelas — e uma chamada. O `02`
ficou como arquivo que só diz para onde o trabalho foi, para quem segue a
ordem antiga não encontrar "arquivo não existe". O `99` lê a coluna com os dois
nomes e compara com a transformação que o store registrou. O P1 ganhou uma
guarda: depois que o `01` novo rodar, o store em disco é do próprio
`dsm_prepare()`, e compará-lo consigo mesmo não provaria nada.

**Não é preciso rodar o `01` de novo agora**: o store em disco é idêntico ao
que ele produziria. O do P1 é uma cópia do mesmo store, com a receita dentro,
e pode ser apagado.


## 2026-09-26 — `has_na`, corrigido pela segunda vez

A primeira correção (contar NA sobre todas as linhas extraídas, não só as que
sobraram do QC) tinha um defeito: um ponto no oceano, onde a pilha inteira é
nodata, contaria como NA em **todos** os canais, e o risco marcaria todos —
sem apontar nenhum. Um ponto sem dado em canal nenhum não diz nada sobre canal
nenhum; o relatório de culpa da regra da janela já separa isso com o
`n_sole_cause`. Agora a contagem, e o percentual, são sobre os pontos que têm
dado em **pelo menos um** canal.

O relatório do console também passa a explicar o `has_na`: cada ponto onde o
canal é NA é um ponto que o QC descarta, e o mapa terá um buraco onde o canal
for nodata. Não diz "descarte-o", como diz para os canais constantes: se o
canal vale os pontos que custa é um julgamento sobre a variável, e o relatório
dá a contagem de que esse julgamento precisa.

**Verificação.** `tests/test_prepare.R` ganhou um ponto fora do raster (p14,
coluna 46 de 40): 14 pontos extraídos, 2 problemas de preditor, 10 depois do
QC — e **nenhum** canal marcado por causa dele (a temperatura continua com 1
NA, em 13 pontos com dado). No P1, o `p1_11` agora aceita o
`channel_risk.csv` diferente **só** no `has_na` — as colunas que descrevem o
canal idênticas, `constant` e `near_constant` exatamente onde estavam,
`has_na` só onde o `01` não dizia nada — e informa quais canais ele passou a
apontar.


## 2026-09-26 — Uma passada só: os centros saem da mesma leitura dos patches

### O que mudou

O `dsm_prepare()` lia cada banda duas vezes. Primeiro o `terra::extract()`,
em série, para os valores de centro de todos os pontos — sobre eles rodam o
QC, a detecção de tipos e o filtro de variância. Depois a extração paralela,
só dos patches dos pontos que sobreviveram. Nestes rasters (faixas de uma
linha, LZW) as duas descomprimem as mesmas linhas, e a primeira, sozinha,
levava ~30 dos 93 minutos do P1 — e levaria horas no conjunto completo de 41
mil pontos.

Agora cada banda é lida **uma vez, em paralelo**, e a leitura devolve as duas
coisas:

- **ponto longe da borda** → os patches e o centro saem da mesma janela;
- **ponto dentro do raster, mas perto demais da borda** para a maior janela →
  uma leitura 1 × 1 só do centro (o ponto fica na tabela de pontos, só não tem
  patch — como antes);
- **ponto fora do raster** → nada é lido, o centro fica NA e o QC o descarta
  — o `terra::extract()` também devolvia NA.

O QC, os tipos e o filtro de variância rodam sobre esses centros, e os arrays
são cortados, **uma vez só, na gravação**, para as linhas da tabela de pontos
e os canais que sobraram. (Cortar logo depois do QC e de novo na regra da
janela copiaria cada janela duas vezes.)

**O custo**: patches lidos para pontos que o QC depois descarta — 9% dos
pontos do SOC. A memória dos arrays passa a ser dimensionada pelos pontos
lidos, não pelos que sobram; o plano de RAM já contava assim.

### Por que o resultado não pode mudar

O centro é a mesma célula que o `terra::extract()` devolvia (a célula que
contém o ponto, pelo `cellFromXY`, que é o que os patches do `02` já usavam),
e passa pelo mesmo `qc_band_values()`, aplicado à mesma leitura. O P1 compara
de novo com o store antigo, ainda em disco: a tabela de pontos, os tipos, o QC
e os patches têm que sair idênticos.

### A alternativa descartada

Converter os rasters para TIFF em blocos (256 × 256): um patch de 15 × 15
tocaria 1 a 4 blocos em vez de 15 linhas inteiras de 160 mil colunas. Seria a
maior aceleração possível, mas é uma decisão sobre os dados de origem, e foi
recusada — os rasters ficam como estão.

### Como isto é verificado

- **`tests/test_prepare.R`**: os centros agora são conferidos contra a fórmula
  do raster para **todos** os pontos da tabela, inclusive os de borda (p5, p6,
  p13), que ganham leitura própria; e a tabela de pontos tem que sair idêntica
  com 2 núcleos, com uma leitura por ponto e com blocos de 3 linhas — não só
  os patches.
- Antes de pedir a execução, a lógica foi simulada em Python, índice por
  índice, nos três planos de leitura do teste (2, 13 e 10 leituras por banda):
  centros e patches idênticos, e todas as contagens conferidas.
- **P1**, de novo contra o store antigo: o `p1_01` informa as leituras por
  banda. Se o P1 falhar, a mensagem agora avisa para **não** rodar o `01`: ele
  chama o `dsm_prepare()` com `overwrite = TRUE` e substituiria o store que o
  P1 usa como referência.


## 2026-09-27 — P1 da passada única: PASS 19/19, e o gargalo agora é o HD

### O resultado

- **19 de 19.** Os patches saíram idênticos bit a bit, e a tabela de pontos
  idêntica valor a valor. O centro lido na passada é a mesma célula que o
  `terra::extract()` devolvia.
- **Tempo total de 92,8 para 68,4 min** (−24,4 min, −26%). A fase serial
  sumiu: sobram 0,2 min fora da extração.
- A extração em si passou de 60,6 para 68,2 min. O modelo abaixo atribui ~2
  min aos 388 pontos a mais que ela agora lê (378 GB contra 367). Os outros
  ~5 min são a diferença entre o que o modelo prevê para a corrida antiga
  (66,2) e o que ela mediu (60,6), e ficam sem explicação. Talvez o cache do
  sistema, aquecido pela fase serial que vinha antes.
- **`has_na` não marca canal nenhum.** Os 13 pontos com problema de preditor
  são nodata na pilha inteira. A primeira correção (99e3a91) teria marcado
  os 174 canais não constantes, cada um com os mesmos 13 NA, e não apontaria
  nenhum. Era exatamente o defeito que a segunda correção tirou, agora visto
  nos dados e não só suposto. O `channel_risk.csv` sai idêntico ao antigo,
  inclusive nas contagens.
- Nenhum ponto do SOC ficou perto da borda nem fora do raster. A leitura 1 × 1
  só de centro não foi exercitada aqui, só no teste.

### Onde está o tempo agora: no HD, não na CPU

O D: é **um HD SATA de 12 TB**. Os 181 rasters somam 1.052 GB comprimidos
(11,6 GB por banda contínua). O C: é NVMe, mas tem 493 GB livres, e os
rasters não cabem.

O `tools/extraction_io_model.py` calcula, banda por banda, os bytes
comprimidos das linhas que o plano de leitura toca, a partir dos cabeçalhos
dos TIFFs, sem ler um pixel. Ajustei dois modelos aos 13 lotes de 15 bandas
que o log do P1 cronometrou:

| modelo | o tempo do lote é proporcional a | R² |
|---|---|---|
| disco | a **soma** dos bytes do lote (um disco dividido) | **0,973** |
| CPU | o **maior** do lote (o núcleo mais lento) | 0,242 |

A vazão fica em ~99 MB/s somando os 15 workers. O último lote tinha uma banda
só (`wtd_annual`, 4,5 GB) e levou ~0,5 min, ou seja **~150 MB/s com um leitor
sozinho** (entre 125 e 190 MB/s, pela resolução do log). Quinze leitores no
mesmo HD rendem menos que um: a cabeça do disco pula entre quinze arquivos.

### O que isso muda, e o que fica para depois

- **Mais núcleos não aceleram a extração.** Menos provavelmente aceleram.
  Quantos, só medindo.
- **Ler só as linhas de que os patches precisam.** Hoje uma leitura cobre da
  primeira à última linha dos pontos do seu grupo de colunas, e as linhas do
  meio são lidas à toa. Cortando os grupos também por linha, o modelo prevê
  378 → 233 GB no dev (68 → ~44 min) e 944 → 763 GB nos 41 mil pontos
  (164 → ~133 min). Isso vale se a vazão se mantiver, o que não é garantido:
  pedaços menores e separados custam mais buscas num HD.
- **Descartado — ordenar as bandas por tamanho nos lotes.** Ajudaria se o
  limite fosse a CPU. Com o disco no limite, o lote custa a soma dos bytes, e
  a ordem não muda a soma. A medição evitou uma otimização errada.
- **Descartado — copiar os rasters para o NVMe:** 1.052 GB não cabem em 493.
- **Tiles:** recusado, é decisão sobre os dados de origem.
- **A extração completa** (41.385 pontos), com o código de hoje: ~2,7 h
  (944 GB lidos).
- **Nos mapas (passo 5) isto pesa mais.** Prever um mapa lê cada raster na área
  inteira. O leitor do `dsm_predict()` tem que ler cada bloco uma vez só e
  aplicar todos os seeds nele, e o número de leitores simultâneos tem que ser
  medido neste HD. Um benchmark de leitores simultâneos serve às duas coisas.
  Fica para o passo 5, e o resultado volta para o `dsm_prepare()` antes da
  extração completa.


## 2026-09-27 — Um lote maior que a dobra não treinava nada, e a unidade passava

Achado ao escrever a grade padrão do passo 2. O carregador de treino descarta
o último lote incompleto (`drop_last = TRUE`, porque o BatchNorm não aceita um
lote de uma amostra). Com um `batch_size` maior que o conjunto de treino da
dobra, **toda época tinha zero passos de gradiente**. A perda de validação da
rede não treinada é finita, então virava a "melhor época", e a unidade saía
com status `success`. A tabela de comparação ranqueava uma rede aleatória ao
lado das treinadas, sem erro nem aviso.

No SOC isso nunca aconteceu (~2.100 pontos de treino por dobra, lotes de até
512). Num conjunto pequeno aconteceria calado, e a grade padrão sorteava
128/256/512 para qualquer conjunto.

Agora o `run_cnn_resample()` recusa a grade **antes da primeira unidade**,
dizendo quais configs e qual o tamanho da menor dobra. O carregador também
recusa, como rede de segurança para qualquer outro caminho.

**Verificação.** Em `tests/test_api_run.R`, um lote de 64 com a menor dobra de
16 pontos é recusado antes de treinar, e nenhum modelo é gravado.


## 2026-09-27 — Passo 2: o `dsm_train()` lê do store o que vinha do exemplo do SOC

`dsm_train(data, tune_length = 30)` é a chamada de um usuário novo, e três
padrões dela vinham do SOC, não dos dados:

| o quê | antes | agora |
|---|---|---|
| janelas da grade | 3/9/15, de qualquer store: um store de 5 e 11 parava com "this grid needs 3, 9, 15" | todas as janelas do store, sozinhas e em pares (o ramo duplo usa duas) |
| tamanhos de lote | 128/256/512, fosse qual fosse a dobra | os que dão **≥ 4 passos por época** na menor dobra |
| a inversa do alvo | `identity`: um store log1p treinado sem `transform = expm1` dava toda métrica "nativa" em escala log | a do store; uma função passada é conferida e recusada se discorda |
| núcleos | o que o device tivesse; os exemplos digitavam 30, e o RF usava todos os 32 lógicos | `n_cores`, com um só significado: NULL = físicos − 1 |

**Detalhes que importam:**

- **Janelas.** Primeiro as janelas sozinhas, em ordem crescente, depois os
  pares em ordem lexicográfica. Para 3/9/15 isso dá **exatamente** a lista que
  o SOC escreveu à mão, na mesma ordem. Por isso a mesma semente sorteia a
  mesma grade de antes (o teste confere).
- **Quatro passos, e por que quatro.** O warmup, o platô de LR e a parada
  antecipada contam em épocas. Uma época de uma atualização transforma
  `patience = 60` em sessenta atualizações, um cronograma que quer dizer outra
  coisa. Quatro é o mínimo que a grade do SOC já usava (512 em ~2.100 pontos),
  então no SOC nada sai. Se nenhuma opção dá quatro, fica a maior potência de
  2 que dá, nunca abaixo de 2 (o BatchNorm precisa de duas linhas).
- **A inversa.** A função passada é conferida em cinco valores contra a do
  store. Uma equivalente escrita de outro jeito (`function(z) exp(z) - 1`)
  passa. O `03` deixou de digitar `expm1`.
- **Threads.** Pertencem à sessão R, não ao device. Um device passado mantém as
  threads com que foi criado, a menos que `n_cores` também seja dado.
  `set_torch_threads()` saiu de dentro do `setup_torch_device()` para isso.
- **O RF** agora usa `n_cores` (antes, `num.threads = 0`: todos os 32 lógicos).
  Não medi se isso muda o tempo dele. O resultado não deveria mudar, porque o
  ranger semeia cada árvore, mas isso também não medi.

**O que não mudou.** Nenhuma grade do SOC muda: o `03` passa a grade
explícita, e mesmo a padrão, com 3/9/15 e ~2.100 pontos, sai igual. Nenhuma
chamada dos exemplos muda de resultado, porque todas passavam `expm1`, que
concorda com o log1p do store.

**Alternativas descartadas.**

- Exigir `windows` no `make_tune_grid()`: quebraria as chamadas diretas. Sem
  `windows` fica o conjunto do SOC, e se o store não tiver essas janelas o
  `dsm_train()` para na hora, dizendo como corrigir.
- Mínimo de um passo por época: evitaria o zero (a correção anterior), mas não
  o cronograma sem sentido.

**Verificação.**

- `tests/test_train_defaults.R`, novo, rápido e sem treino. Cobre as opções de
  janela, a grade do SOC reproduzida sob a mesma semente, a grade de um store
  5/11, os lotes (com o SOC intacto), o carregador que recusa, a inversa em
  seis casos, e `n_cores` recusado na porta.
- `tests/test_api_run.R`, seção 3b. A grade padrão treina num store de uma
  janela, pede só a janela 3, usa lotes que cabem na menor dobra, e deixa o
  torch em 1 thread depois de `n_cores = 1`. As previsões gravadas são
  exatamente `expm1` da saída da rede. O `clamp` fica aberto nesse teste,
  senão uma rede com saídas todas negativas daria 0 com qualquer inversa, e o
  teste passaria no vazio.


## 2026-09-27 — T1: quantas threads o torch deve usar nesta máquina

Todo script de treino digitava `setup_torch_device(n_threads = 30)`. A máquina
tem 32 núcleos **lógicos**, mas só 16 **físicos**. A regra do framework é
físicos − 1 = 15, porque hyperthreads dividem as unidades aritméticas de um
núcleo e o torch em CPU costuma ficar mais lento quando tem mais threads que
núcleos. Ninguém mediu qual das duas vale para esta rede nesta máquina. A
extração acabou de mostrar que a intuição "mais núcleos, mais rápido" falhou
aqui.

O `_t1_threads_benchmark.R` mede **segundos por época** de duas configs da
rede real, no store de dev, com 5, 7, 15 e 30 threads:

- `heavy_3x15`: ramo duplo 3 + 15, três blocos, flatten, SE. É a forma mais
  cara que a grade sorteia.
- `light_3`: um ramo 3×3, dois blocos, gap. É barata, e nela domina o custo
  fixo de cada operação.

**Como mede, e por quê:**

- **Um processo R por contagem**, lançado com `OMP_NUM_THREADS` já definido. O
  OpenMP dimensiona o pool quando o torch carrega. O B6 suspeitava justamente
  disso na diferença de 1,2e-4 de CCC do seu subprocesso.
- **Duas passadas em ordens opostas** (15, 30, 7, 5 e de volta), porque a
  máquina deriva ao longo do tempo. A diferença entre as passadas é o ruído, e
  uma diferença menor que ele não é resultado.
- **12 épocas na pesada e 60 na leve.** O executor grava o tempo arredondado a
  0,01 min (0,6 s), e dez épocas da leve dariam alguns por cento só de
  arredondamento.
- **5 e 7 respondem a uma segunda pergunta:** dividir os 15 núcleos em 3
  unidades de 5 threads, ou 2 de 7, lado a lado. O refit final com N seeds
  (passo 3) são N unidades independentes. A tabela **estima** isso. Só uma
  rodada real de unidades lado a lado confirmaria, porque elas dividem a
  banda de memória.
- A mesma semente em todas as contagens. Assim o resultado também mostra se
  uma rodada se reproduz com a mesma contagem, e se a contagem muda os
  números. Isso é relatado, não checado, porque não decide nada sobre
  velocidade.

**O que decide.** O `n_cores` que os scripts de treino passam. Se 15 estiver
dentro do ruído do mais rápido, os scripts largam o 30 e usam o padrão. Se não
estiver, passam o número medido, com esta rodada como motivo.

**Custo:** 8 processos, ~15–25 min, com a máquina parada. Resultado: pendente.


## 2026-09-27 — T1: PASS 3/3 — 30 threads não ganham; o ganho está em dividir os núcleos

Antes, a suíte inteira: **26/26 em 3,7 min**, com o `test_train_defaults.R`
novo (39/39) e a seção 3b do `test_api_run.R`. O passo 2 está verificado.

### O resultado

Segundos por época, média das duas passadas. O maior desvio entre passadas
foi 5,7% (30 threads, config pesada); nos outros casos, até 3%.

| threads | `heavy_3x15` | vs 15 | `light_3` | vs 15 |
|---|---|---|---|---|
| 5  | 5,58 | 0,67× | 0,67  | 1,04× |
| 7  | 4,75 | 0,78× | 0,67  | 1,04× |
| 15 | 3,72 | 1     | 0,695 | 1     |
| 30 | 3,50 | 1,06× | 0,78  | 0,89× |

### O que diz

1. **Hyperthreading não paga.** Com 30 threads a config pesada ficou 6%
   mais rápida (no limite do ruído) e a leve ficou 11% mais lenta. Numa grade
   que mistura as duas, a diferença fica em torno de 3%. O critério escrito
   antes da rodada ("15 dentro do ruído do mais rápido?") deu veredito
   dividido: não na pesada (6,4% contra 5,7% de ruído), sim na leve.
2. **Uma unidade sozinha usa mal a máquina.** Na pesada, triplicar as threads
   (5 → 15) deu só 1,5× de velocidade. Na leve, nada acima de 5.
3. **Por isso a estimativa de unidades lado a lado:** 3 unidades de 5 threads
   renderiam ~2,0× (pesada) e ~3,1× (leve) a vazão de uma unidade de 15.
   **É estimativa:** supõe que as unidades não disputam a banda de memória e
   o cache L3, e só uma rodada real confirma.
4. **Reprodutibilidade.** Com a mesma contagem, as duas passadas deram
   `val_ccc` idêntico bit a bit (8 de 8), então o treino é determinístico dado
   o par (seed, threads). Entre contagens, a pesada variou **0,067 de CCC** em
   12 épocas e a leve, 1e-6.
   - Na pesada, a contagem de threads age como uma troca de seed: outra ordem
     de soma nas reduções leva a outra trajetória.
   - Na leve, as operações são pequenas demais para o torch dividir, e nada
     muda (é também por isso que ela não acelera).
   - Em 12 épocas o `val_ccc` ainda sobe rápido, então 0,067 não é o efeito no
     fim do treino; só mostra que o efeito existe. Isso também explica os
     1,2e-4 do B6.

### O que decide, e o que não

- **O 30 dos scripts fica, por enquanto.** Entre 15 e 30 a diferença é de ~3%
  numa grade mista. Trocar mudaria os números de toda unidade treinada daqui
  em diante, como uma troca de seed. Melhor mudar **uma vez só**, para o
  esquema de unidades em paralelo, se ele se confirmar.
- **A contagem de threads por unidade faz parte do que torna um resultado
  reprodutível**, e o executor ainda não a registra. Vai para o passo 3, junto
  com o registro da rodada final.
- **Próximo: T2.** Três unidades de 5 threads lado a lado contra as mesmas três
  em sequência com 15, mesmo trabalho, medindo o tempo de parede. Se
  confirmar ~2×, o `dsm_final()` (passo 3) treina as N seeds assim, com threads
  por unidade fixas e registradas. Assim o resultado não depende de quantas
  unidades couberam na máquina.


## 2026-09-27 — T2: unidades lado a lado, medidas em vez de estimadas

O T1 estimou que 3 unidades de 5 threads lado a lado renderiam ~2× a vazão de
uma de 15, supondo que elas não se atrapalham. Mas elas dividem a banda de
memória e o cache L3, e só rodando juntas dá para saber quanto. Isso importa
porque o modelo final são N unidades independentes: a config escolhida
reajustada com N seeds, no `dsm_final()` do passo 3.

O `_t2_parallel_units.R` treina **as mesmas 6 unidades** (a config pesada do
T1, seeds 42 a 47, 12 épocas, sem parada antecipada) de três jeitos:

| arranjo | processos × threads | unidades por processo |
|---|---|---|
| `1x15` | 1 × 15 | 6 em sequência (o que o `04` faz hoje) |
| `3x5`  | 3 × 5  | 2 |
| `2x7`  | 2 × 7  | 3 |

**Como mede, e por quê:**

- **Uma barreira.** Cada processo carrega o store, marca que está pronto e
  espera os outros. Sem ela, os três leriam 1,3 GB cada do mesmo HD ao mesmo
  tempo, e a comparação seria sobre o disco, não sobre o treino.
- **O cache da dobra é montado dentro do tempo medido**, por cada processo,
  porque é isso que o `dsm_final()` vai fazer: tensores do torch não passam de
  um processo R para outro.
- **Duas passadas em ordens opostas**, como no T1.
- **A checagem que mais importa (t2_04).** A seed 42 com 5 threads, treinada
  ao lado de outras duas unidades, precisa dar exatamente o `val_ccc` que deu
  sozinha no T1, e o mesmo vale com 7 e com 15 threads. Se rodar lado a lado
  mudasse o número (um OpenMP que encolhe o pool sob carga faria isso), o
  resultado dependeria de quantas unidades couberam na máquina, e o desenho
  paralelo cairia, fosse qual fosse a velocidade.

**Custo:** ~20–30 min. Cada processo apaga os checkpoints que gravou (36 ×
~49 MB) depois de salvar os tempos.


## 2026-09-27 — T2: PASS 4/4 — lado a lado é 1,52× mais rápido e não muda nenhum número

### O resultado

As mesmas 6 unidades, tempo de parede a partir da barreira, média das duas
passadas:

| arranjo | tempo | vs `1x15` | atraso por unidade causado pelos vizinhos | pico de RAM por processo |
|---|---|---|---|---|
| `1x15` (sequência) | 4,58 min | 1 | — | 10,3 GB |
| `2x7` | 3,44 min | **1,33×** | 1,12× | 10,0 GB |
| `3x5` | 3,02 min | **1,52×** | 1,25× | 10,0 GB |

As passadas diferiram em no máximo 0,06 no ganho.

- **t2_04: lado a lado não muda nenhum número.** A seed 42 deu, ao lado de
  outras unidades, exatamente o que deu sozinha no T1: 0,363685 com 5
  threads, 0,360795 com 7 e 0,379778 com 15. E t2_03: as 18 combinações
  (seed, threads) se repetiram bit a bit entre as passadas. **O resultado de
  uma unidade depende só de (seed, threads por unidade)**, não de quem roda ao
  lado nem de quantas unidades couberam na máquina.
- **O ganho medido (1,52×) é menor que o estimado (2,0×)**, porque os vizinhos
  atrasam cada unidade em ~25% (banda de memória e cache). A estimativa do T1
  estava na direção certa e superestimava; foi para isso que o T2 existiu.
- **Um achado que ninguém procurava: ~10 GB de pico por processo**, com 1,26
  GB de janelas carregadas (3 e 15), ou seja ~8×. Não depende das threads. No
  dev isso deixa rodar 3 processos (30 GB de 63). No conjunto completo (41 mil
  pontos, janelas ~11× maiores) um processo só, na mesma proporção, passaria
  de 100 GB. **Isso precisa ser resolvido antes da rodada completa**, com
  paralelo ou sem: o `04` de hoje também não caberia.

### O que decide para o `dsm_final()`

1. **Unidades lado a lado, com threads por unidade FIXAS** (padrão 5,
   medido aqui). O número de processos simultâneos sai de `n_cores ÷ threads
   por unidade`, limitado pela RAM. Como o t2_04 provou que os vizinhos não
   mudam os números, esse limite só muda o tempo, nunca o resultado.
2. **Sempre em subprocesso, com `OMP_NUM_THREADS` definido antes de o torch
   carregar**, mesmo com um processo só. Assim o resultado não depende do
   estado da sessão do usuário. Era exatamente essa a suspeita do B6 para os
   seus 1,2e-4.
3. **As threads por unidade entram no registro da rodada**, como a seed.
4. **O limite de RAM é estimado pelo T2** (1,5 GB + 7 × o tamanho das janelas
   carregadas), e o pico real de cada processo é medido e relatado, para a
   estimativa poder ser conferida em cada rodada.


## 2026-09-27 — Passo 3: `dsm_final()`, e a declaração de cada hiperparâmetro da CNN escolhida

`R/final.R` é o estágio `04` com o conjunto de dados retirado. Recebe um
`dsm_fit` (ou a pasta de uma rodada de tuning) e o `dsm_data`, escolhe a
config, reajusta com N seeds e grava **os mesmos arquivos nos mesmos
lugares**, de modo que o `05` lê uma rodada do `dsm_final()` como lê uma do
`04`.

**O que ele faz, na ordem:**

1. **A escolha**, com as regras do `04`, sem mudança: one_se por padrão, o
   motivo registrado quando cai para rank 1, o aviso quando a margem é menor
   que o ruído entre seeds, `freeze_selection()` na rodada de tuning.
2. **O recorte do refit** pelo critério do próprio plano de tuning
   (`refit_split()`).
3. **A escala dos preditores**, calculada uma vez, entregue a todos os
   processos e gravada ao lado dos pesos.
4. **As seeds lado a lado**, conforme o T1 e o T2:
   - cada seed roda num processo R próprio, lançado com `OMP_NUM_THREADS` já
     definido, mesmo quando há um processo só;
   - as threads por seed são fixas (padrão 5) e ficam registradas;
   - quantos processos rodam juntos sai de `n_cores ÷ threads`, limitado pela
     RAM (a estimativa do T2 fica ao lado do pico medido);
   - cada processo "reserva" a próxima seed com um `dir.create()`, que é
     atômico: dois processos nunca treinam a mesma seed, e a distribuição se
     ajusta sozinha quando uma seed para antes da outra.
5. **De N seeds ao que o mapa precisa**: mediana do ensemble, conformal com os
   resíduos da CV do tuning (o achado do `04`: a validação do refit é uma
   dobra só e dava 83,6% de cobertura para 90% nominais) e fator de smearing.
6. **Retomada:** uma seed com registro de sucesso e checkpoint no disco não é
   treinada de novo.

**A declaração pedida** ("declarar cada parâmetro ótimo da CNN que foi
selecionada") sai no console, em `final_report.md` e em
`selected_hyperparameters.csv`. Para cada hiperparâmetro vêm o valor, **se foi
a busca que o escolheu ou se a grade o fixou**, os valores que a grade tentou
e o que ele significa. A distinção importa: um valor só é "o ótimo" se a
grade oferecia alternativas. Um parâmetro fixo foi decidido por quem escreveu
a grade, não pelos dados. O relatório também diz como a config foi escolhida
(regra, métrica, média ± sd, posição, quantas configs ficaram dentro de um
erro-padrão, o ruído entre seeds), como foi o refit e o resultado no teste
(por seed e do ensemble, cobertura do conformal, fator de smearing).

**Duas coisas que o `04` fazia e o `dsm_final()` não faz:**

- **O intervalo normalizado pela dispersão entre seeds.** O próprio `04`
  explicava por que ele não vale quando a calibração vem da CV. O intervalo
  normalizado de verdade (nível + dissimilaridade) é do `dsm_predict()`, no
  passo 4.
- **Treinar na sessão do usuário.** É isso que torna o resultado reproduzível
  seja qual for a sessão.

**Verificação:**

- **`tests/test_final.R`**, na suíte lenta, sobre 96 pontos em 8 sítios.
  Confere o layout, a seleção congelada na rodada de tuning e a declaração (o
  único parâmetro que a grade variou aparece como "tuned", com os valores
  tentados). Confere também a retomada sem retreino e cinco argumentos
  recusados antes de treinar. Acima de tudo: **2 processos de 1 thread e 1
  processo de 1 thread dão métricas por seed idênticas bit a bit**, a
  propriedade do T2 mantida pela suíte.
- **P3** (`_p3_final_assembly_check.R`) roda a montagem do `dsm_final()` sobre
  os arquivos por seed que o `04` deixou na rodada implantada e compara tudo
  com o que o `04` gravou: ensemble, quantil conformal de 90 e 95%, fator de
  smearing, métricas por seed e resumo. Retreinar não serviria para isso: o
  `04` treinou com 30 threads e o `dsm_final()` treina com 5, e o T1 mostrou
  que a contagem muda os números. Sobre os mesmos arquivos, a comparação tem
  que ser exata (até 1e-12, por causa do leitor de CSV).

**Uma ferramenta nova:** `tools/r_balance.py` confere o balanço de `()`, `[]`
e `{}` com números de linha reais, entendendo strings, comentários e raw
strings. O `r_skeleton.py` descarta as linhas de comentário e desalinha a
numeração. Os 80 arquivos R do projeto passam.

**O que ainda não muda:** o `04` continua sendo o script dele. Trocá-lo por uma
chamada ao `dsm_final()` só depois do P3, e é uma decisão do usuário, porque
um novo modelo final treinado com 5 threads por seed tem números diferentes
(estatisticamente equivalentes) do implantado, treinado com 30.


## 2026-09-27 — `dsm_final()` verificado: suíte 27/27, P3 7/7; e a declaração do modelo implantado

**Suíte: 27/27 em 4,3 min.** O `test_final.R` passou 22/22 na primeira
execução real do `dsm_final()`, com os processos filhos do `callr`, a reserva
por `dir.create()` e a retomada. **2 processos de 1 thread e 1 processo de 1
thread deram as seeds idênticas bit a bit.**

**P3: 7/7.** A montagem do `dsm_final()` sobre os arquivos por seed da rodada
implantada (`final_20260918_150311`, cfg_003, 10 seeds) reproduz o que o
`04` gravou:

- o ensemble;
- os quantis conformais: 39,62 t/ha em 90% e 55,33 t/ha em 95%, idênticos;
- o fator de smearing 1,346057, idêntico;
- as métricas de teste das 10 seeds e o resumo, dentro de 1e-12.

**A declaração do modelo implantado.** Ele foi ajustado pelo `04`, antes de o
`dsm_final()` existir, então não tinha relatório. Retreinar para ter um mudaria
as seeds (T1). O `dsm_report_final()` escreve a declaração a partir dos
arquivos, com a mesma montagem que o P3 acabou de provar, e **só acrescenta**
`final_report.md` e `selected_hyperparameters.csv` à pasta da rodada. O que o
`04` não registrou (threads, cronograma do refit) o relatório diz que não foi
registrado, em vez de supor. O `04b_final_report.R` faz isso para o SOC. O
`test_final.R` ganhou a mesma situação simulada (uma rodada "do `04`", sem
registros): a declaração sai igual à que o `dsm_final()` escreveu.

**O que a declaração vai mostrar sobre o SOC:** a rodada de tuning de onde o
modelo saiu é a de dev, com `tune_length = 3`, então a "busca" comparou três
configurações. A tabela vai dizer quais valores foram de fato comparados, e
isso é o que ela existe para dizer.


## 2026-09-27 — Passo 4 começa: o intervalo "nível + DI", e o custo real de um mapa a 250 m

### O intervalo com escala ajustada

`conformal_scaled_calibrate()` / `conformal_scaled_interval()`, em
`R/conformal.R`:

- **Ajuste da escala:** metade dos pontos de calibração ajusta
  σ = a + b·nível + c·DI por mínimos quadrados sobre |resíduo|.
- **Calibração do q:** a outra metade calibra q sobre |resíduo|/σ, com a
  correção (n+1).
- **O intervalo:** predição ± q·σ.

A separação em metades é o que mantém a garantia de cobertura (split conformal
com escore ponderado localmente: Papadopoulos et al. 2008; Lei et al. 2018,
§5.2). Se o ajuste e o quantil usassem os mesmos pontos, o ajuste absorveria
os resíduos que o quantil deveria medir. A escala tem um piso de 5% da mediana
de |resíduo|: um ajuste linear pode ir a zero na borda das covariáveis, e um
intervalo de largura zero afirmaria uma certeza que os dados nunca deram.

**Testes** (`test_conformal.R`, 11 novos): num conjunto em que o erro cresce
com o nível **e** com o DI, por construção:

- as duas escalas ajustadas cobrem 90% ± 3% no total;
- nível + DI deixa a cobertura uniforme nos dois eixos, com menos de metade da
  variação da constante;
- só nível deixa o eixo do DI desigual;
- a garantia vale em média com n = 60 (30 para ajustar, 30 para calibrar);
- a escala nunca cai abaixo do piso;
- três usos errados são recusados.

### A referência do DI virou função

`aoa_reference()` e `aoa_di()`, em `R/aoa.R`, trazem a construção que morava
dentro do `07`: o pixel central de cada ponto que treinou ou validou, com QC e
a escala **do modelo**, e o fold em que foi deixado de fora. O DI de validação
cruzada de cada ponto (distância ao vizinho mais próximo fora do seu fold) é
o DI com que vem cada resíduo de calibração. O mapa, a AOA e o intervalo
passam a medir contra a mesma referência: duas cópias de "o que o modelo viu"
seriam o único jeito de este intervalo perder a garantia sem erro algum,
calibrado com um DI e aplicado com outro. O `test_aoa.R` ganhou 4 checagens:
um ponto da referência tem DI 0, o DI de CV é o do limiar, o teste fica fora e
uma escala permutada é recusada.

### U1: qual intervalo o mapa leva

`_u1_conformal_level_di.R` calibra os três intervalos de 90% (constante, só
nível, nível + DI) nos resíduos da CV do cfg_003 e os mede nos 591 pontos de
teste. Mostra a cobertura no total, por quinto do nível, por quinto do DI e
dentro/fora da AOA, com a largura ao lado. Um intervalo que cobre por ser
enorme não melhorou nada. Resultado: pendente.

### O custo real de um mapa a 250 m

A rodada de dev do `05` (grade de ~20 km, 358 mil pixels válidos) prevê a
**164 pixels válidos por segundo**, com 10 seeds. A leitura foi desprezível
(23 s contra 2.184 s de inferência). A 250 m, com ~22% da grade válida, seriam
~2,3 bilhões de pixels: **~4 meses de máquina**. O custo não é o disco: a
rede recalcula as convoluções de cada patch 15×15 do zero para cada pixel, e
patches vizinhos compartilham 14 de 15 colunas.

Rodar a rede como **totalmente convolucional** (as convoluções uma vez sobre a
faixa inteira; depois, por pixel, o pooling e a cabeça) cortaria isso em
~100×, para ~1–2 dias. Mas isso só é **exato** quando o ramo usa padding
`valid` e não tem bloco SE. O zero-padding de cada patch e o SE por patch não
são invariantes à translação. O cfg_003 implantado (um ramo 15×15,
`valid_large`, sem SE) é exato. Isso é uma decisão, não um detalhe, e fica
para o usuário.
