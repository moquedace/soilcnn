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

`predict_raster_dir` (env var `SOC_PREDICT_RASTER_DIR`) remaps the raster paths
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
