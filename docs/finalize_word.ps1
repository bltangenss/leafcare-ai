# Word арқылы: (1) сызықтық форматтағы формулаларды Word теңдеулеріне (OMML) айналдыру,
# (2) мазмұн (TOC) өрістерін жаңарту, (3) .docx және .pdf ретінде сақтау
param([string]$Path)
$full = (Resolve-Path $Path).Path
$pdf = [System.IO.Path]::ChangeExtension($full, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($full)
    # Теңдеу абзацтары: "\t<формула>\t(n.m)"
    $targets = @()
    foreach ($p in $doc.Paragraphs) {
        $t = $p.Range.Text
        if ($t -match "^\t(.+)\t\(\d+\.\d+\)\r?$" -and $p.Range.OMaths.Count -eq 0) {
            $s = $p.Range.Start + 1
            $targets += ,@($s, ($s + $Matches[1].Length))
        }
    }
    [array]::Reverse($targets)
    $n = 0
    foreach ($tg in $targets) {
        $r = $doc.Range($tg[0], $tg[1])
        $om = $r.OMaths.Add($r)
        $om.OMaths.Item(1).BuildUp()
        $n++
    }
    "Теңдеулер: $n"
    $doc.Fields.Update() | Out-Null
    foreach ($t in $doc.TablesOfContents) { $t.Update() | Out-Null }
    $doc.Repaginate()
    foreach ($t in $doc.TablesOfContents) { $t.UpdatePageNumbers() | Out-Null }
    "Беттер саны: " + $doc.ComputeStatistics(2)
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Close()
    "PDF: $pdf"
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
