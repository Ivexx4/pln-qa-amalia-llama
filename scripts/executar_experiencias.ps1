<#
.SYNOPSIS
  Corre o plano de experiências RAG para o relatório (execuções completas, com todas as perguntas do tema).

.DESCRIPTION
  Fases (por esta ordem):
    sweeps      chunks-topk (6), prompt-minscore (6), overlap (2) -> 14 execuções com o corpus das sementes
    corpus      corpus expandido por ligações com max-links 5, 10 e 20 -> 3 execuções (as outras variáveis ficam nas omissões)
    comparar    comparação de cada execução com a baseline sem RAG (IC bootstrap e testes)
    consolidar  junta tudo em results\meta3\consolidado.csv e tabela_relatorio.md

  É retomável: voltar a correr continua onde parou (as caches já calculadas não são repetidas).
  NÃO alterar ficheiros de qa_pipeline\ durante as experiências: o manifesto regista o hash do código e
  recusa retomar uma execução com código diferente.

.EXAMPLE
  .\scripts\executar_experiencias.ps1 -Ensaio      # 2 perguntas, saída em results\ensaio: estima tempos
  .\scripts\executar_experiencias.ps1              # plano completo
  .\scripts\executar_experiencias.ps1 -Fase comparar,consolidar
#>
param(
    [ValidateSet("sweeps", "corpus", "comparar", "consolidar", "tudo")]
    [string[]]$Fase = @("tudo"),
    [string]$Tema = "Cibersegurança",
    [switch]$Ensaio
)

$py = ".\.venv\Scripts\python.exe"
function Quero($nome) { return ($Fase -contains $nome) -or ($Fase -contains "tudo") }
function Slug($t) {
    $s = $t.Normalize([Text.NormalizationForm]::FormD) -replace "\p{M}", ""
    return ($s.ToLower() -replace "[^a-z0-9]+", "-").Trim("-")
}

$saida = if ($Ensaio) { "results\ensaio" } else { "results" }
$extra = if ($Ensaio) { @("--limit", "2", "--output-dir", $saida) } else { @() }
$pasta = "data\rag\$(Slug $Tema)"
$resultados = New-Object System.Collections.Generic.List[object]
$relogio = [System.Diagnostics.Stopwatch]::StartNew()

New-Item -ItemType Directory -Force logs | Out-Null
Start-Transcript -Path "logs\experiencias-$(Get-Date -Format 'yyyyMMdd-HHmmss').log" | Out-Null

function Executar([string]$Nome, [string[]]$Argumentos) {
    Write-Host "`n=== $Nome ===" -ForegroundColor Cyan
    Write-Host "rag $($Argumentos -join ' ')" -ForegroundColor DarkGray
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    & $py main.py rag @Argumentos
    $codigo = $LASTEXITCODE
    $estado = if ($codigo -eq 0) { "OK" } else { "FALHOU" }
    Write-Host ("-> {0} em {1:mm\:ss} (código {2})" -f $estado, $t.Elapsed, $codigo) -ForegroundColor $(if ($codigo -eq 0) { "Green" } else { "Red" })
    $resultados.Add([pscustomobject]@{ Passo = $Nome; Estado = $estado; Duracao = $t.Elapsed.ToString("hh\:mm\:ss") })
}

# --- verificações prévias -------------------------------------------------------------------------------
try { Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null }
catch { Write-Host "Ollama não responde em localhost:11434. Inicie-o antes (ollama serve)." -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
if (-not (Test-Path "$pasta\corpus.jsonl")) {
    Write-Host "Corpus inexistente: a recolher as sementes." -ForegroundColor Yellow
    Executar "corpus (sementes)" @("corpus", "--theme", $Tema)
}
if ($Ensaio) { Write-Host "MODO ENSAIO: 2 perguntas, saída em $saida (não polui results\meta3)." -ForegroundColor Yellow }

# --- 1. varrimentos com o corpus das sementes -----------------------------------------------------------
if (Quero "sweeps") {
    foreach ($s in "sweep_chunks_topk", "sweep_prompt_minscore", "sweep_overlap") {
        Executar $s (@("sweep", "--theme", $Tema, "--sweep", "configs\rag\$s.json") + $extra)
    }
}

# --- 2. corpus expandido por ligações (cada corpus fica guardado para a execução ser reprodutível) -------
if (Quero "corpus") {
    Copy-Item "$pasta\corpus.jsonl" "$pasta\corpus.sementes.jsonl" -Force
    Copy-Item "$pasta\fontes.csv"   "$pasta\fontes.sementes.csv"   -Force
    foreach ($ml in 5, 10, 20) {
        $guardado = "$pasta\corpus.expandido-$ml.jsonl"
        if (Test-Path $guardado) {
            Copy-Item $guardado "$pasta\corpus.jsonl" -Force
            Write-Host "Corpus expandido ($ml) reposto de $guardado" -ForegroundColor Yellow
        } else {
            Executar "corpus expandido (max-links $ml)" @("corpus", "--theme", $Tema, "--expand-links", "--max-links", "$ml", "--min-links", "2")
            Copy-Item "$pasta\corpus.jsonl" $guardado -Force
        }
        Executar "all corpus-expandido-$ml" (@("all", "--theme", $Tema, "--run-id", "corpus-expandido-$ml") + $extra)
    }
    Copy-Item "$pasta\corpus.sementes.jsonl" "$pasta\corpus.jsonl" -Force   # repõe o corpus das sementes
    Copy-Item "$pasta\fontes.sementes.csv"   "$pasta\fontes.csv"   -Force
    Write-Host "Corpus das sementes reposto." -ForegroundColor Yellow
}

# --- 3. comparação com a baseline sem RAG ---------------------------------------------------------------
if (Quero "comparar") {
    $execucoes = Get-ChildItem "$saida\meta3" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "^(chunks-topk|prompt-minscore|overlap|corpus)-" }
    foreach ($e in $execucoes) {
        Executar "comparar $($e.Name)" (@("compare", "--theme", $Tema, "--run-id", $e.Name) + $extra)
    }
}

# --- 4. consolidação -------------------------------------------------------------------------------------
if (Quero "consolidar") {
    Write-Host "`n=== consolidar ===" -ForegroundColor Cyan
    & $py scripts\consolidar_resultados.py --meta3 "$saida\meta3"
    $resultados.Add([pscustomobject]@{ Passo = "consolidar"; Estado = $(if ($LASTEXITCODE -eq 0) { "OK" } else { "FALHOU" }); Duracao = "" })
}

Write-Host "`n================ RESUMO ($($relogio.Elapsed.ToString('hh\:mm\:ss'))) ================" -ForegroundColor Cyan
$resultados | Format-Table -AutoSize
$falhas = @($resultados | Where-Object { $_.Estado -eq "FALHOU" }).Count
Write-Host "$($resultados.Count) passos, $falhas falhas." -ForegroundColor $(if ($falhas) { "Red" } else { "Green" })
if ($Ensaio) { Write-Host "Multiplique a duração média de uma execução 'all' por 17 (14 dos varrimentos + 3 de corpus) para estimar o plano completo." -ForegroundColor Yellow }
Stop-Transcript | Out-Null
