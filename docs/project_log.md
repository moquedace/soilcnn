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

## Pendente

| etapa | o quê |
|---|---|
| 5 | `one_se()` (o piso de ruído já é medido pelo `seed_noise_floor()`) |
| 6 | predição: inverter loops → 5 sementes → FCN |
| 7 | promover predição a função; `_scratch_1km_test/` some |
| 8 | paralelismo sobre configs (medir antes) |
| 9 | virar pacote |
| 10 | importância de variáveis |
