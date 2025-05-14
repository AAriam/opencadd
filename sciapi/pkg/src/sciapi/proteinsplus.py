"""[ProteinsPlus](https://proteins.plus) webserver API.

ProteinsPlus is a web portal for structural analysis of macromolecules,
offering a variety of tools for processing structural data,
such as protonation/tautomerization state prediction and binding pocket detection.

Notes
-----
The API endpoints are subject to rate limiting (30 jobs/minute).
Inidividual rate limits for some of the tools
with heavy CPU/RAM usage do exist (e.g., for DoGSiteScorer).

References
----------
- [About ProteinsPlus](https://proteins.plus/pages/about)
- [ProteinsPlus REST API](https://proteins.plus/help/index#REST-help)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gzip
import csv
import re
import io
from xml.etree import ElementTree as ET
import re

import pylinks as pl

from sciapi import util

if TYPE_CHECKING:
    from typing import Sequence, Any, Literal


class ProteinsPlusAPI:
    """ProteinsPlus web server API."""
    def __init__(
        self,
        base_url: str = "https://proteins.plus/api",
        status_num_tries: int = 5,
        status_sleep_init: int = 1,
        status_sleep_scale: int = 2,
        response_num_tries: int = 500,
        response_sleep_init: int = 10,
        response_sleep_scale: int = 1,
        retry_status_codes: Sequence[int] = (408, 429, 500, 502, 503, 504),
    ):
        self._base_url = pl.url.create(base_url)
        self._retry_config = pl.http.HTTPRequestRetryConfig(
            status_codes_to_retry=retry_status_codes,
            config_status=pl.http.RetryConfig(
                num_tries=status_num_tries,
                sleep_time_init=status_sleep_init,
                sleep_time_scale=status_sleep_scale,
            ),
            config_response=pl.http.RetryConfig(
                num_tries=response_num_tries,
                sleep_time_init=response_sleep_init,
                sleep_time_scale=response_sleep_scale,
            ),
        )
        return

    def dogsite(
        self,
        pdb_id: str,
        chain_id: str | None = None,
        ligand_id: str | tuple[str, str, int] | None = None,
        include_subpockets: bool = True,
        calculate_druggability: bool = True,
        algorithm: Literal["scorer", "3"] = "3",
        ligand_bias: bool = False,
    ) -> DoGSiteResponse:
        """Detect binding pockets for a protein structure using DoGSite algorithms.

        Parameters
        ----------
        pdb_id
            PDB ID of the protein; either valid 4-letter PDB ID, or dummy ID issued by Proteins.Plus
            for an uploaded PDB file (see `upload_pdb`).
        chain_id
            Chain ID of a polymer instance in the PDB file,
            so that binding pockets are only detected for that chain.
            if not provided (i.e. when set to `None`; default), pockets are detected for all chains.
        ligand_id
            Identifier for a ligand instance in the PDB file, to calculate its coverage for each pocket.
            The identifiers are: ligand ID (het ID), chain ID, and residue number of the specific instance.
            This can be given as either a tuple (e.g., `("w32", "A", 1101)`),
            or a string where the three identifiers are joined by underscores (e.g., `"w32_A_1101"`).
            If not provided (i.e. when set to `None`; default), coverage data will not be calculated.
        include_subpockets
            Whether to divide detected pockets into sub-pockets and return both (True; default),
            or to only return pockets (False).
        calculate_druggability
            Whether to calculate druggability scores for each (sub)pocket (True; default).
        algorithm
            Algorithm to use for binding site detection:
            - "scorer": DoGSiteScorer
            - "3": DoGSite3 (newer version)
        ligand_bias
            Whether the grid should be biased by the coordinates od the selected ligand.
            This is only used by the DoGSite3 algorithm
            and only when `ligand_id_chain_num` is provided.

        Returns
        -------
        DoGSiteScorerResponse
            A Future-like object that holds the job URL
            and can be used to retrieve the results once the job is done.

        References
        ----------
        - [DoGSiteScorer API](https://proteins.plus/help/dogsite_rest)
        """

        params = {
            "pdbCode": pdb_id,
            "analysisDetail": str(int(include_subpockets)),
            "bindingSitePredictionGranularity": str(int(calculate_druggability)),
            "ligand": self._convert_ligand_id(ligand_id),
            "chain": chain_id if chain_id is not None else "",
        }
        if algorithm == "3":
            params["ligandBias"] = str(int(ligand_bias))
        json_key = "dogsite" if algorithm == "scorer" else "dogsite3"
        base_response = self.submit_job(
            endpoint="dogsite_rest" if algorithm == "scorer" else "dogsite3_rest",
            json={json_key: params},
        )
        return DoGSiteResponse(base_response.job_url, retry_config=self._retry_config, algorithm=algorithm)

    def poseedit(self, pdb_id: str, ligand: str | tuple[str, str, int]) -> PoseEditResponse:
        """Calculate ligand interaction diagram.

        Parameters
        ----------
        pdb_id
            PDB ID of the protein; either valid 4-letter PDB ID,
            or dummy ID issued by Proteins.Plus for an uploaded PDB file (see `upload_pdb`).
        ligand
            Identifier for a ligand instance in the PDB file.
            The identifiers are: ligand ID (het ID), chain ID, and residue number of the specific instance.
            This can be given as either a tuple (e.g., `("w32", "A", 1101)`),
            or a string where the three identifiers are joined by underscores (e.g., `"w32_A_1101"`).

        Returns
        -------
        PoseEditResponse
            A Future-like object that holds the job URL
            and can be used to retrieve the results once the job is done.
        """
        base_response = self.submit_job(
            endpoint="poseview2_rest",
            json={"poseview2": {"pdbCode": pdb_id, "ligand": self._convert_ligand_id(ligand)}},
        )
        return PoseEditResponse(base_response.job_url, retry_config=self._retry_config)

    def protoss(self, pdb_id: str) -> ProtossResponse:
        """Predict protonation and tautomerization, and add missing hydrogen atoms.

        Parameters
        ----------
        pdb_id
            PDB ID of the protein; either valid 4-letter PDB ID,
            or dummy ID issued by Proteins.Plus for an uploaded PDB file (see `upload_pdb`).
        retry_config
            Retry configurations for HTTP requests.

        Returns
        -------
        ProtossResponse
            A Future-like object that holds the job URL
            and can be used to retrieve the results once the job is done.
        """
        base_response = self.submit_job(
            endpoint="protoss_rest",
            json={"protoss": {"pdbCode": pdb_id}},
        )
        return ProtossResponse(base_response.job_url, retry_config=self._retry_config)

    def upload_pdb(self, content: bytes | str) -> PDBUploadResponse:
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
            base_response = self.submit_job(
                endpoint="pdb_files_rest",
                files={"pdb_file[pathvar]": ("dummy_name.pdb", file)},
                headers=None,
            )
        return PDBUploadResponse(base_response.job_url, retry_config=self._retry_config)

    def submit_job(
        self,
        endpoint: str,
        files: dict[str, Any] | None = None,
        headers: dict[str, str] | Literal["default"] | None = "default",
        json: dict[str, Any] | None = None,
    ) -> ProteinsPlusResponse:
        """Submit a job to the ProteinsPlus web server.

        This method is not meant to be called directly,
        unless you want to submit a custom job to the server,
        or if you want to use an endpoint that is not yet implemented in this class.

        Parameters
        ----------
        endpoint
            API endpoint to submit the job to.
            For example, "dogsite_rest" for DoGSiteScorer.
        json
            JSON payload to send to the server.

        Returns
        -------
        URL of the job to be used to retrieve the results.
        """
        headers = {
            "Content-type": "application/json", "Accept": "application/json"
        } if headers == "default" else headers
        job_url = pl.http.request(
            url=self._base_url / endpoint,
            verb="POST",
            json=json,
            files=files,
            headers=headers,
            response_type="json",
            response_verifier=lambda response_dict: "location" in response_dict.keys(),
            retry_config=self._retry_config,
        )["location"]
        return ProteinsPlusResponse(job_url, retry_config=self._retry_config)

    @staticmethod
    def _convert_ligand_id(ligand_id: str | tuple[str, str, int] | None) -> str:
        """Create a ProteinsPlus-compatible ligand ID.

        Parameters
        ----------
        ligand_id
            Ligand ID as a string or a tuple of (het ID, chain ID, residue number).

        Returns
        -------
        Ligand ID as a string.
        """
        if not ligand_id:
            return ""
        if isinstance(ligand_id, str):
            return ligand_id
        return "_".join([str(i) for i in ligand_id])


class ProteinsPlusResponse:
    """ProteinsPlus web server response.

    This is a Future-like object that holds the job URL
    and can be used to retrieve the results once the job is done.
    Calling any of the methods or properties of this class
    (except for `job_url`) will block until the results are available.
    A job usually takes around 1-2 minutes to complete.

    Parameters
    ----------
    job_url
        URL of the job.
    """

    def __init__(
        self,
        job_url: str,
        retry_config: pl.http.HTTPRequestRetryConfig,
        response_verifier: callable = lambda response_dict: response_dict["status_code"] == 200,
    ):
        self._job_url = job_url
        self._retry_config = retry_config
        self._response_verifier = response_verifier
        self._job_results: dict[str, Any] = {}
        return

    @property
    def job_url(self) -> str:
        """URL of the job."""
        return self._job_url

    @property
    def job_results(self) -> dict[str, str]:
        """Get the results when the job is done.

        Notes
        -----
        The job is done when the 'status_code' key
        in the JSON response is 200
        (instead of 202, which means still processing).
        """
        if self._job_results:
            return self._job_results
        self._job_results = pl.http.request(
            url=self._job_url,
            verb="GET",
            response_type="json",
            response_verifier=self._response_verifier,
            retry_config=self._retry_config,
        )
        return self._job_results


class DoGSiteResponse(ProteinsPlusResponse):
    """DoGSite binding site detection results.

    This is a Future-like object that holds the job URL
    and can be used to retrieve the results once the job is done.
    Calling any of the methods or properties of this class
    (except for `job_url`) will block until the results are available.
    A job usually takes around 1-2 minutes to complete.

    Parameters
    ----------
    job_url
        URL of the job.

    References
    ----------
    - [DoGSiteScorer API](https://proteins.plus/help/dogsite_rest)
    """

    _REGEX_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
    r"""Match any number; positive or negative; decimal or integer.

    This regular expression matches numbers that may or may not have a minus sign at the beginning,
    and may or may not have a decimal point with one or more digits after it.
    Breakdown:
    * '^': Matches the start of the string.
    * '-?': Matches an optional minus sign.
    * '\d+': Matches one or more digits (0-9).
    * '(?:\.\d+)?': Matches an optional decimal point followed by one or more digits.
    The '?:' syntax creates a non-capturing group.
    * '$': Matches the end of the string.
    """

    def __init__(
        self, job_url: str,
        retry_config: pl.http.HTTPRequestRetryConfig,
        algorithm: Literal["scorer", "3"],
    ):
        super().__init__(job_url, retry_config)
        self._algorithm = algorithm
        self._full_data: list[dict[str, Any]] | None = None
        return

    @property
    def full_data(self) -> list[dict[str, Any]]:
        """Complete data of the binding sites.

        This is a convenience property that parses and combines
        all data returned by the API into a single list of dictionaries.
        Each dictionary contains the data for a single pocket,
        corresponding to a row in the result table, with the addtional
        key 'mrc', which contains the binary data of the CCP4/MRC map file
        corresponding to the pocket.

        When the calculations are done with DoGSiteScorer,
        the following additional keys are also added to each dictionary:

        center_x, center_y, center_z
            Coordinates of the center of the pocket extracted from the PDB file.
        max_radius
            Maximum radius of the pocket extracted from the PDB file.
        atom_serials
            Serial numbers of the atoms in the pocket extracted from the PDB file.

        DoGSite3 does not include the pocket center and radius
        in the PDB files returned by the `residues` endpoint.
        It also resets the serial numbers of the atoms in each PDB file,
        so they do not match the original PDB file.
        Therefore, for DoGSite3, instead of the above keys,
        only a 'pdb' key is added to each dictionary,
        which contains the content of the PDB file.

        To get the pocket center and radius for DoGSite3,
        you need to extract them from the mrc file.
        Similarly, atom serial numbers can be obtained by
        matching the ATOM records in the returned PDB files
        with the original PDB.
        """
        if self._full_data:
            return self._full_data
        result_table = self.result_table
        ccp4_files = self.pockets
        pdb_files = self.residues
        raw_dicts = csv.DictReader(io.StringIO(result_table), delimiter='\t')
        full_data: list[dict] = []
        for raw_pocket_data, ccp4_file, pdb_file in zip(raw_dicts, ccp4_files, pdb_files):
            pocket_data = util.recursive_type_cast(raw_pocket_data)
            if self._algorithm == "scorer":
                pocket_center, pocket_radius, pocket_atom_serial_numbers = self._parse_residues(pdb_file)
                pocket_data["center_x"], pocket_data["center_y"], pocket_data["center_z"] = pocket_center
                pocket_data["max_radius"] = pocket_radius
                pocket_data["atom_serials"] = pocket_atom_serial_numbers
            else:
                pocket_data["pdb"] = pdb_file
            pocket_data["mrc"] = ccp4_file
            full_data.append(pocket_data)
        self._full_data = full_data
        return self._full_data

    @property
    def result_table(self) -> str:
        """Result table containing general binding site data.

        This is a string containing the result table in TSV (tab-separated values) format.
        The table contains the following columns:

        name
            Name of the pocket, as annotated by DoGSiteScorer.
        lig_cov
            Percentage of ligand volume covered by the pocket.
        poc_cov
            Percentage of pocket volume covered by the ligand.
        lig_name
            PDB name, chain ID, and residue number of the ligand in the PDB file,
            for which `lig_cov` and `poc_cov` are calculated.
        volume
            Pocket volume in cubic angstroms (Å^3), calculated from number of grid points.
        enclosure
            Ratio of number of surface to hull grid points.
        surface
            Pocket surface in square angstroms (Å^2), calculated from number of grid points.
        depth
            Depth of the pocket in angstroms (Å).
        surf/vol
            Surface to volume ratio.
        lid/hull
            (Probably) ratio of number of surface to hull grid points.
            This is usually set to "-".
        ellVol
            Ellipsoid volume.
            This is usually set to "-".
        ell c/a
            Ellipsoid main axis ratio c/a (with a > b > c).
        ell b/a
            Ellipsoid main axis ratio b/a (with a > b > c).
        siteAtms
            Number of surface atoms lining the pocket.
        accept
            Number of hydrogen-bond acceptor atoms.
        donor
            Number of hydrogen-bond donor atoms.
        hydrophobic_interactions
            Number of hydrophobic contacts.
        hydrophobicity
            Hydrophobicity of the pocket (apparently in range [0, 1]),
        metal
            Number of metal atoms.
        Cs
            Number of carbon atoms.
        Ns
            Number of nitrogen atoms.
        Os
            Number of oxygen atoms.
        Ss
            Number of sulfur atoms.
        Xs
            Number of other atoms.
        negAA
            Percentage of negatively charged amino acids.
        posAA
            Percentage of positively charged amino acids.
        polarAA
            Percentage of polar amino acids.
        apolarAA
            Percentage of apolar amino acids.
        simpleScore
            Simple druggability score (in range [0, 1]), based on a linear combination
            of volume, hydrophobicity and enclosure values.
        drugScore
            Druggability score (in range [0, 1]), predicted by a support vector machine (libsvm)
            trained on a subset of meaningful descriptors.

        In addition, the table contains one column for each of
        the 20 amino acid 3-letter codes (e.g. ALA, ARG, etc.),
        containing the number of these residues in the pocket.
        """
        return pl.http.request(
            url=self.job_results["result_table"],
            response_type="str",
        )

    @property
    def descriptor_explanation(self) -> str:
        """Explanation of the columns in the result table."""
        return pl.http.request(
            url=self.job_results["descriptor_explanation"],
            response_type="str",
        )

    @property
    def pockets(self) -> tuple[bytes]:
        """CCP4/MRC map data of the pockets.

        This is a tuple of bytes objects,
        each containing the binary data of a CCP4/MRC map file
        corresponding to a pocket.
        The order of the map files corresponds to
        the order of the pockets in the result table.
        """
        return tuple(
            gzip.decompress(pl.http.request(url=url_file,response_type="bytes"))
            for url_file in self.job_results["pockets"]
        )

    @property
    def residues(self) -> tuple[str]:
        """PDB files of residues in the pockets.

        This is a tuple of strings,
        each containing the content of a PDB file
        corresponding to a pocket.
        The order of the PDB files corresponds to
        the order of the pockets in the result table.

        These are pseudo-PDB files,
        containing three pieces of information:
        1. The center of the pocket (x, y, z) coordinates.
        2. The maximum radius of the pocket.
        3. ATOM records of the atoms in the pocket.
        """
        return tuple(
            pl.http.request(
                url=url_file,
                response_type="str",
            )
            for url_file in self.job_results["residues"]
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Parameters used for the job.

        This is a dictionary containing
        the parameters used for the job:

        chain
            Chain IDs of the polymer instances (or "all")
            in the PDB file that were analyzed.
        bindingSitePredictionGranularity
            For example, "Properties && druggability"
        analysisDetail
            For example, "Pocket(s) && subpocket(s)"
        """
        return self.job_results["parameters"]

    def _parse_residues(
        self,
        pdb_file: str,
    ) -> tuple[tuple[float, float, float], float, tuple[int]]:
        """Parse a PDB file returned by the `residues` endpoint of DoGSiteScorer.

        Note that this only applies to DoGSiteScorer,
        and not to DoGSite3.
        DoGSite3 does not include the pocket center and radius
        in the PDB files returned by the `residues` endpoint.
        It also resets the serial numbers of the atoms in each PDB file,
        so they do not match the original PDB file.

        The returned files are pseudo-PDB files,
        containing three pieces of information:
        1. The center of the pocket (x, y, z) coordinates.
        2. The maximum radius of the pocket.
        3. Macromolecule atoms making up the pocket.

        Parameters
        ----------
        pdb_file
            PDB file content.

        Returns
        -------
        A 3-tuple containing:
        - Pocket center (x, y, z) coordinates.
        - Pocket radius.
        - serial numbers of atoms in the pocket.

        Notes
        -----
        The PDB file starts with 6 HEADER records,
        followed by ATOM records for each atom in the pocket.
        The 5 first HEADER records contain no useful information,
        while the 6th HEADER record contains the pocket center and radius.
        The ATOM records are identical to the original PDB file,
        so it suffices to only extract their serial numbers.

        Example PDB file content:
        ```
        HEADER	Output of DoGSiteScorer by A. Volkamer
        HEADER	Pocket 0 with 131 binding site atoms written.
        HEADER	References:
        HEADER	A. Volkamer et al. Analyzing the topology of active sites: on the prediction of pockets and subpockets. J. Chem. Inf. Model. 2010,50(11), 2041-52
        HEADER	A. Volkamer et al. Combining global and local measures for structure-based druggability predictions. J. Chem. Inf. Model. 2012,52,360-372
        HEADER	Geometric pocket center at  15.91  32.33  11.03 with max radius 12.42
        ATOM    149  CB  LEU A 718      17.938  39.850  15.721  0.00  0.00           C
        ATOM    150  CG  LEU A 718      16.428  40.166  15.737  0.00  0.00           C
        ```
        """
        lines = pdb_file.splitlines()
        info_line = lines[5]
        center_and_radius = tuple(
            float(elem) for elem in info_line.split() if self._REGEX_NUM.match(elem)
        )
        pocket_center = center_and_radius[:3]
        pocket_radius = center_and_radius[3]
        pocket_atom_serial_numbers = tuple(int(line[6:11]) for line in lines[6:])
        return pocket_center, pocket_radius, pocket_atom_serial_numbers


class PoseEditResponse(ProteinsPlusResponse):
    """PoseEdit ligand interaction diagram results.

    This is a Future-like object that holds the job URL
    and can be used to retrieve the results once the job is done.
    Calling any of the methods or properties of this class
    (except for `job_url`) will block until the results are available.
    A job usually takes around 1-2 minutes to complete.

    Parameters
    ----------
    job_url
        URL of the job.

    References
    ----------
    - [PoseEdit API](https://proteins.plus/help/poseview2_rest)
    """

    def __init__(self, job_url: str, retry_config: pl.http.HTTPRequestRetryConfig):
        super().__init__(job_url, retry_config)
        return

    @property
    def data(self) -> dict:
        """Input parameters for the PoseEdit JavaScript library.

        These are used for the generation of the 2D-diagram."""
        return pl.http.request(
            url=self.job_results["result_json"],
            response_type="json",
        )

    @property
    def svg(self) -> str:
        """PoseEdit interaction diagram in SVG format."""
        svg_str = pl.http.request(
            url=self.job_results["result_svg"],
            response_type="str",
        )
        # The SVG file has elements that only look good on a white background,
        # but it has no background color.
        # To fix this, we add a white rectangle as the background.
        return self._add_svg_background(svg_str)

    @staticmethod
    def _add_svg_background(svg_str: str, color: str = "#fff") -> str:
        """Add a background rectangle to an SVG string.

        Parameters
        ----------
        svg_str
            The SVG content as a string.
        color
            The background color to apply (default is light grey '#f0f0f0').

        Returns
        -------
        Modified SVG string with background rectangle added.

        Raises
        -------
        ValueError
            If viewBox or size attributes are missing and can't determine size.
        """
        # Parse SVG string
        try:
            root = ET.fromstring(svg_str)
        except ET.ParseError as e:
            raise ValueError(f"Invalid SVG content: {e}")

        # Handle namespaces
        ns_match = re.match(r'\{.*\}', root.tag)
        ns = ns_match.group(0) if ns_match else ''

        # Get viewBox or width/height
        viewBox = root.attrib.get('viewBox')
        if viewBox:
            x, y, width, height = map(float, viewBox.split())
        else:
            width = root.attrib.get('width')
            height = root.attrib.get('height')
            if width and height:
                # Strip units (e.g., '600pt', '600px')
                width = float(re.sub(r'[a-zA-Z]+', '', width))
                height = float(re.sub(r'[a-zA-Z]+', '', height))
                x, y = 0.0, 0.0
            else:
                raise ValueError("SVG missing viewBox and width/height attributes.")

        # Create background rect element
        rect = ET.Element(f'{ns}rect', {
            'x': str(x),
            'y': str(y),
            'width': str(width),
            'height': str(height),
            'fill': color
        })

        # Insert rect after <defs> if present, else as first child
        insert_index = 1 if len(root) > 0 and root[0].tag.endswith('defs') else 0
        root.insert(insert_index, rect)

        # Return modified SVG string
        return ET.tostring(root, encoding='unicode')


class ProtossResponse(ProteinsPlusResponse):
    """Protoss protonation state prediction results.

    This is a Future-like object that holds the job URL
    and can be used to retrieve the results once the job is done.
    Calling any of the methods or properties of this class
    (except for `job_url`) will block until the results are available.
    A job usually takes around 1-2 minutes to complete.

    Parameters
    ----------
    job_url
        URL of the job.

    References
    ----------
    - [Protoss API](https://proteins.plus/help/protoss_rest)
    """

    def __init__(self, job_url: str, retry_config: pl.http.HTTPRequestRetryConfig):
        super().__init__(job_url, retry_config)
        return

    @property
    def protein(self) -> str:
        """PDB file of the protein with predicted protonation states.

        This is a string containing the content of the PDB file
        with the predicted protonation states.
        """
        return pl.http.request(
            url=self.job_results["protein"],
            response_type="str",
        )

    @property
    def ligands(self) -> str:
        """PDB file of the ligand with predicted protonation states.

        This is a string containing the content of the PDB file
        with the predicted protonation states.
        """
        return pl.http.request(
            url=self.job_results["ligands"],
            response_type="str",
        )

    @property
    def log(self) -> str:
        """Log file of the job.

        This is a string containing the content of the log file
        with the predicted protonation states.
        """
        return pl.http.request(
            url=self.job_results["log"],
            response_type="str",
        )


class PDBUploadResponse(ProteinsPlusResponse):
    """ProteinsPlus PDB upload response."""

    def __init__(self, job_url: str, retry_config: pl.http.HTTPRequestRetryConfig):
        super().__init__(
            job_url,
            retry_config=retry_config,
            response_verifier=lambda response_dict: bool(response_dict.get("id")),
        )
        return

    @property
    def dummy_pdb_id(self) -> str:
        """Dummy PDB ID assigned to the uploaded PDB file."""
        return self.job_results["id"]
