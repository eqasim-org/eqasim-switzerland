import requests
from tqdm import tqdm
import subprocess as sp
import os.path

def configure(context, require):
    require.stage("utils.java")
    require.config("matsim_url", "https://github.com/matsim-org/matsim/releases/download/matsim-0.10.1/matsim-0.10.1.zip")

def execute(context):
    url = context.config["matsim_url"]
    target_path = "%s/matsim.zip" % context.cache_path

    r = requests.get(url, stream = True)
    total = int(r.headers["content-length"])

    with tqdm(desc = "Downloading MATSim", total = total) as progress:
        with open(target_path, 'wb+') as f:
            for chunk in r.iter_content(chunk_size = 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

    sp.check_call([
        "unzip", "matsim.zip"
    ], cwd = context.cache_path)

    assert(os.path.exists("%s/matsim-0.10.1/matsim-0.10.1.jar" % context.cache_path))

    class_path = [
        "%s/matsim-0.10.1/libs/*" % context.cache_path,
        "%s/matsim-0.10.1/matsim-0.10.1.jar" % context.cache_path
    ]

    return class_path
