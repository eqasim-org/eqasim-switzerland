# eqasim Java runtime

`matsim.runtime.eqasim` builds the Switzerland JAR from
`https://github.com/eqasim-org/eqasim-java.git` by default:

```yaml
eqasim_branch: cmdp_new_developments
eqasim_commit: -1
```

`-1` (an integer or string) selects the latest commit on the configured branch.
Normal Synpp validation queries the remote branch commit and rebuilds the
runtime when it changes. This requires access to the Git repository, including
when checking a cached runtime. Debug mode can bypass cache validation.

Set `eqasim_commit` to a commit hash or tag to pin a revision instead. The
configured branch must exist; cloning retains its full history so older commits
on that branch remain available for checkout.

Providing `eqasim_path` uses that existing JAR instead of cloning or building.
Its modification time controls cache validation; no remote Git check is made.
