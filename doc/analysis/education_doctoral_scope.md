# Doctoral medical-field and population coverage audit

`analysis.education_doctoral_scope` follows the usual `configure`/`execute`
structure and writes to `<analysis_path>/education_doctoral_scope`. It reads
processed structural-survey reports and the frozen BFS UH enrolment cube.
The snapshot content hash invalidates only this analysis when its source changes.
It changes no model status, degree, employment, enrolment reference or locations.

The BFS cube has an actual programme field and supports the exact medical-field
exclusion used by the linked employment publication. Pharmacy remains included.
The supplied survey extract has no current doctoral subject-field variable.
Occupation and employer/school NOGA evidence are exported for inspection, with
the current study field explicitly unavailable. These proxies must not be
treated as verified medical/non-medical programmes or automatic correction rules.

Outputs preserve annual weights, student flags, country status, employment,
ongoing-versus-completed doctoral study and reported commuting codes. This makes
unknown school destinations, possible reporting conflicts and healthcare work
visible without deleting valid respondents. Missing education commuting is not
treated as absence of doctoral enrolment.

The least invasive next action is to reconcile BFS institution enrolment with the
survey's permanent-resident/private-household frame. Keep the all-field BFS total
and the medical split as separate references. A matching number after a medical
exclusion is not evidence that the entire survey gap consists of medical students.
