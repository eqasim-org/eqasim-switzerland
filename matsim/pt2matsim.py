import requests
from tqdm import tqdm
import subprocess as sp
import os.path

def configure(context, require):
    require.config(
        "pt2matsim_url",
        "https://bintray.com/polettif/matsim/download_file?file_path=org%2Fmatsim%2Fpt2matsim%2F18.7%2Fpt2matsim-18.7-shaded.jar"
    )

def execute(context):
    url = context.config["pt2matsim_url"]
    target_path = "%s/pt2matsim.jar" % context.cache_path

    r = requests.get(url, stream = True)
    total = int(r.headers["content-length"])

    with tqdm(desc = "Downloading pt2matsim", total = total) as progress:
        with open(target_path, 'wb+') as f:
            for chunk in r.iter_content(chunk_size = 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

    sp.check_call([
        "java", "-cp", "pt2matsim.jar", "org.matsim.pt2matsim.run.CreateDefaultOsmConfig", "test_config.xml"
    ], cwd = context.cache_path, stdout = sp.PIPE, stderr = sp.PIPE)

    assert(os.path.exists("%s/test_config.xml" % context.cache_path))

    return target_path
