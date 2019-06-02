import argparse
import importlib
import time
import datetime
import os
import requests
import shutil
import tqdm


def download(url, path, fname, redownload=False):
    """
    Downloads file using `requests`. If ``redownload`` is set to false, then
    will not download tar file again if it is present (default ``False``).
    """
    outfile = os.path.join(path, fname)
    download = not os.path.isfile(outfile) or redownload
    print("[ downloading: " + url + " to " + outfile + " ]")
    retry = 5
    exp_backoff = [2 ** r for r in reversed(range(retry))]

    pbar = tqdm.tqdm(unit='B', unit_scale=True, desc='Downloading {}'.format(fname))

    while download and retry >= 0:
        resume_file = outfile + '.part'
        resume = os.path.isfile(resume_file)
        if resume:
            resume_pos = os.path.getsize(resume_file)
            mode = 'ab'
        else:
            resume_pos = 0
            mode = 'wb'
        response = None

        with requests.Session() as session:
            try:
                header = {'Range': 'bytes=%d-' % resume_pos,
                          'Accept-Encoding': 'identity'} if resume else {}
                response = session.get(url, stream=True, timeout=5, headers=header)

                # negative reply could be 'none' or just missing
                if resume and response.headers.get('Accept-Ranges', 'none') == 'none':
                    resume_pos = 0
                    mode = 'wb'

                CHUNK_SIZE = 32768
                total_size = int(response.headers.get('Content-Length', -1))
                # server returns remaining size if resuming, so adjust total
                total_size += resume_pos
                pbar.total = total_size
                done = resume_pos

                with open(resume_file, mode) as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except requests.exceptions.ConnectionError:
                retry -= 1
                pbar.clear()
                if retry >= 0:
                    print('Connection error, retrying. (%d retries left)' % retry)
                    time.sleep(exp_backoff[retry])
                else:
                    print('Retried too many times, stopped retrying.')
            finally:
                if response:
                    response.close()
    if retry < 0:
        raise RuntimeWarning('Connection broken too many times. Stopped retrying.')

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeWarning('Received less data than specified in ' +
                                 'Content-Length header for ' + url + '.' +
                                 ' There may be a download problem.')
        move(resume_file, outfile)

    pbar.close()


def move(path1, path2):
    """Renames the given file."""
    shutil.move(path1, path2)


def untar(path, fname, deleteTar=True):
    """
    Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print('unpacking ' + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cn', action='store_true')
    args = parser.parse_args()
    if args.cn:
        url = 'https://cloud.tsinghua.edu.cn/f/8b26f8d5aa2145c987cc/?dl=1'
    else:
        url = 'https://www.dropbox.com/s/906zxht1p1tsjj3/data.tar.gz?dl=1'
    fname = 'data.tar.gz'

    download(url, '.', fname)
    untar('.', fname)