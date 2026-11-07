param(
 [Parameter(Mandatory=$true)][string]$Workspace,
 [Parameter(Mandatory=$true)][string]$NodeExecutable,
 [Parameter(Mandatory=$true)][string]$NodeModules,
 [Parameter(Mandatory=$true)][string]$PythonExecutable,
 [Parameter(Mandatory=$true)][string]$SkillDirectory,
 [switch]$ResumeDraft
)
$ErrorActionPreference='Stop'
$repository=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if($ResumeDraft){
 $buildDirectory=(Resolve-Path -LiteralPath $Workspace).Path
 if(!(Test-Path -LiteralPath (Join-Path $buildDirectory '.candidate.pptx'))){throw 'No staged draft to resume'}
 $evidence=Get-Content -LiteralPath (Join-Path $buildDirectory 'slide-evidence.json') -Raw | ConvertFrom-Json
 if($evidence.Count -ne 30){throw 'Expected30slide evidence entries'}
 if((Get-ChildItem -LiteralPath (Join-Path $buildDirectory 'previews') -Filter 'slide-*.png').Count -ne 30){throw 'Expected30rendered previews before resuming'}
}else{
 if(Test-Path -LiteralPath $Workspace){throw 'Use a new workspace directory to preserve earlier builds'}
 $buildDirectory=(New-Item -ItemType Directory -Path $Workspace).FullName
 New-Item -ItemType Directory -Path (Join-Path $buildDirectory 'output') | Out-Null
 New-Item -ItemType Junction -Path (Join-Path $buildDirectory 'node_modules') -Target (Resolve-Path -LiteralPath $NodeModules).Path | Out-Null
 Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build_defense_deck.mjs') -Destination (Join-Path $buildDirectory 'build.mjs')
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'finalize_defense_deck.mjs') -Destination (Join-Path $buildDirectory 'finalize.mjs')
$env:RSSR_PRESENTATION_SKILL=(Resolve-Path -LiteralPath $SkillDirectory).Path
$env:RSSR_ARTIFACT_PYTHON=(Resolve-Path -LiteralPath $PythonExecutable).Path
$env:RUNTIME_NODE_MODULES=(Resolve-Path -LiteralPath $NodeModules).Path
$env:RSSR_DRAFT_ONLY='1'
if(!$ResumeDraft){
 & $NodeExecutable (Join-Path $buildDirectory 'build.mjs') $repository $buildDirectory
 if($LASTEXITCODE -ne 0){throw 'Authoring runtime returned nonzero. Inspect the staged draft/previews; use -ResumeDraft only after reviewing them. No failed authoring run is marked passed.'}
}
$candidate=Join-Path $buildDirectory '.candidate.pptx'
$workbooks=Join-Path $buildDirectory 'chart-workbooks.pptx'
& $PythonExecutable (Join-Path $SkillDirectory 'container_tools/materialize_literal_chart_workbooks.py') $candidate $workbooks --workspace $buildDirectory --receipt (Join-Path $buildDirectory 'chart-workbooks.json')
if($LASTEXITCODE -ne 0){throw 'Chart workbook snapshot failed'}
$office=Join-Path $buildDirectory 'office.pptx'
$ppt=New-Object -ComObject PowerPoint.Application
$prior=$ppt.Presentations.Count
try{
 $document=$ppt.Presentations.Open($workbooks,-1,0,0)
 $document.SaveCopyAs($office,24)
 $document.Close()
}finally{if($prior -eq 0){$ppt.Quit()}}
$consistent=Join-Path $buildDirectory 'consistent.pptx'
& $PythonExecutable (Join-Path $PSScriptRoot 'refresh_slide_chart_caches.py') $office $consistent
if($LASTEXITCODE -ne 0){throw 'Chart-cache consistency check failed'}
& $NodeExecutable (Join-Path $buildDirectory 'finalize.mjs') $buildDirectory $consistent
if($LASTEXITCODE -ne 0){throw 'Deck validation failed'}
Write-Output (Join-Path $buildDirectory 'output/RSSR_Defense_Presentation.pptx')
