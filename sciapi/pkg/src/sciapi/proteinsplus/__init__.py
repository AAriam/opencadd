"""[ProteinsPlus](https://proteins.plus) webserver API.

ProteinsPlus is a web portal for structural analysis of macromolecules,
offering a variety of tools for processing structural data,
such as protonation/tautomerization state prediction and binding pocket detection.

References
----------
- [About ProteinsPlus](https://proteins.plus/pages/about)
- [ProteinsPlus REST API](https://proteins.plus/help/index#REST-help)
"""

from __future__ import annotations

import io

import pylinks as pl


URL = pl.url.create("https://proteins.plus/api")


def upload_pdb(
    content: bytes | str,
    retry_config: pl.http.HTTPRequestRetryConfig = pl.http.HTTPRequestRetryConfig(),
) -> str:
    """Upload a custom PDB file to ProteinsPlus.

    The uploaded PDB file will be assigned a dummy PDB ID,
    which can then be used in place of a real PDB ID
    in other ProteinsPlus tools that accept a PDB ID.

    Parameters
    ----------
    content
        Content of the PDB file.
    retry_config
        Retry configurations for HTTP requests.

    Returns
    -------
    Dummy PDB ID to be used in other ProteinsPlus services.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    # Upload the file.
    # This returns a JSON object with a "location" key
    # pointing to the URL to retrieve the dummy PDB ID from.
    with io.BytesIO(content) as file:
        url_of_pdb_id = pl.http.request(
            url=URL / "pdb_files_rest",
            verb="POST",
            files={"pdb_file[pathvar]": ("dummy_name.pdb", file)},
            response_type="json",
            response_verifier=lambda response_dict: "location" in response_dict.keys(),
            retry_config=retry_config,
        )["location"]
    # Get the dummy PDB ID from the retrieval URL.
    # The response must be a JSON object with an "id" key.
    dummy_pdb_id = pl.http.request(
        url=url_of_pdb_id,
        verb="GET",
        response_type="json",
        response_verifier=lambda response_dict: "id" in response_dict.keys(),
        retry_config=retry_config,
    )["id"]
    return dummy_pdb_id
