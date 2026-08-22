# Waiting for a background job, and why pgrep is the wrong way

Three chained jobs failed to fire tonight, each for the same reason, and the
reason is not obvious.

A tool call that writes a script with a heredoc runs as

    bash -c 'cat > scripts/foo.sh <<SHEOF
    ... the whole script text ...
    SHEOF
    nohup bash scripts/foo.sh &'

so that shell's **command line contains the entire text of the script**, and
it stays alive as the parent of the nohup'd job. Anything that greps process
command lines for a string that appears *inside* a script -- a driver's name,
a path, a flag -- matches that shell.

    pgrep -f repin_steps_gpu7        # matches a shell that merely mentions it

`repin_steps_gpu7.sh` finished at 03:11:49. The job waiting for it was still
waiting at 03:31, because a shell whose command line quoted the name was
alive. The same pattern silently disarmed the final regeneration gate and the
router re-fit.

## What to do instead

Wait on **markers on disk**. A driver that writes `$TMP/<step>.done` after
each step gives a count that cannot be confused with anything else:

    for _ in $(seq 1 900); do
      [ "$(ls "$TMP"/*.done 2>/dev/null | wc -l)" -ge "$N" ] && break
      sleep 60
    done

If a process really must be watched, match on something that cannot appear in
a script's text -- a pidfile written by the process itself.

## The other rule from the same night

Never edit a shell script that is running. Bash reads a script by byte offset
and resumes where it left off; inserting a line shifts every offset after it
and the next thing it executes is the middle of a different line. Write a new
file and run that.
