"""
Enhanced SFR2 parsing utilities for FloPy.

This module provides improved parsing functions for the MODFLOW SFR2 package
that handle:
- Better whitespace handling (multiple spaces, tabs)
- MODFLOW-USG node-based models
- More robust error handling
- Version-aware parsing (MODFLOW-2005/NWT vs MODFLOW-USG)

Based on analysis of MODFLOW Fortran source code (gwf2sfr7.f) which uses
free-format reading (READ(In, *)) for all input parsing.
"""

import warnings
from typing import List, Optional, Tuple, Union, Any

import numpy as np


def line_strip(line: str) -> str:
    """
    Remove comments and replace commas from input text for a free formatted
    MODFLOW input file. Also handles tabs and normalizes whitespace.

    Parameters
    ----------
    line : str
        A line of text from a MODFLOW input file

    Returns
    -------
    str
        Line with comments removed and commas/tabs replaced with spaces
    """
    # Remove comments (supports multiple comment styles)
    for comment_flag in [";", "#", "!!", "!"]:
        idx = line.find(comment_flag)
        if idx != -1:
            line = line[:idx]

    # Strip whitespace and replace commas/tabs with spaces
    line = line.strip()
    line = line.replace(",", " ")
    line = line.replace("\t", " ")

    return line


def line_parse(line: str) -> List[str]:
    """
    Convert a line of text into a list of values.

    This handles free-format MODFLOW input where values can be separated by:
    - Single or multiple spaces
    - Tabs
    - Commas

    This is equivalent to Fortran's free-format READ(In, *) behavior.

    Parameters
    ----------
    line : str
        Line of text to parse

    Returns
    -------
    list of str
        List of whitespace-separated values
    """
    line = line_strip(line)
    # split() without arguments splits on any whitespace and removes empty strings
    return line.split()


def pop_item(line: List[str], dtype: type = float, default: Any = None,
             name: str = None) -> Any:
    """
    Pop and convert the next item from a line list.

    This provides improved error handling compared to the original FloPy
    implementation, with optional variable naming for better error messages.

    Parameters
    ----------
    line : list
        List of string values to pop from
    dtype : type
        Type to convert to (float, int, or str)
    default : any, optional
        Default value if list is empty or conversion fails.
        If None, uses dtype(0) for numeric types or empty string for str.
    name : str, optional
        Variable name for better error messages

    Returns
    -------
    Converted value or default
    """
    if default is None:
        if dtype == str:
            default = ""
        else:
            default = dtype(0)

    if not line:
        return default

    try:
        value = line.pop(0)
        if dtype == int:
            # Handle strings like '-10.' by converting to float first
            return int(float(value))
        elif dtype == float:
            return float(value)
        else:
            return dtype(value)
    except (ValueError, IndexError) as e:
        if name:
            warnings.warn(f"Could not parse {name}, using default {default}: {e}")
        return default


def get_next_line(f) -> str:
    """
    Get the next non-blank line from a file.

    Skips blank lines and lines that are only whitespace.

    Parameters
    ----------
    f : file handle
        Open file handle

    Returns
    -------
    str
        Next non-empty line (stripped of trailing whitespace)
    """
    while True:
        line = f.readline()
        if not line:  # EOF
            return ""
        line = line.rstrip()
        if len(line) > 0:
            return line


def isnumeric(s: str) -> bool:
    """
    Check if a string represents a numeric value.

    Parameters
    ----------
    s : str
        String to check

    Returns
    -------
    bool
        True if string can be converted to float
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def get_dataset(line: str, template: List) -> List:
    """
    Parse a line into a dataset using a template for expected values.

    Interprets numbers with decimal points as floats, others as integers.

    Parameters
    ----------
    line : str
        Line to parse
    template : list
        Template list with default values (defines expected length and types)

    Returns
    -------
    list
        Parsed values filling template
    """
    dataset = template.copy()
    values = line_parse(line)

    for i, s in enumerate(values):
        if i >= len(dataset):
            break
        if not isnumeric(s):
            break
        try:
            # Detect if value has decimal point
            if '.' in s or 'e' in s.lower():
                dataset[i] = float(s)
            else:
                dataset[i] = int(s)
        except ValueError:
            break

    return dataset


def parse_1c(line: str, reachinput: bool, transroute: bool,
             version: str = "mf2005") -> Tuple:
    """
    Parse Dataset 1c for SFR2 package.

    This version-aware function handles differences between:
    - MODFLOW-2005/NWT: nstrm < 0 OR transroute triggers irtflg reading
    - MODFLOW-USG: only transroute triggers irtflg reading (not nstrm < 0)

    Parameters
    ----------
    line : str
        Line read from SFR package input file
    reachinput : bool
        Whether REACHINPUT option is active
    transroute : bool
        Whether TRANSROUTE option is active
    version : str
        Model version ('mf2005', 'mfnwt', 'mfusg')

    Returns
    -------
    tuple
        All variables for Dataset 1c (17 values)
    """
    na = 0
    values = line_parse(line)

    # Required parameters
    nstrm = int(values.pop(0))
    nss = int(values.pop(0))
    nsfrpar = int(values.pop(0))
    nparseg = int(values.pop(0))
    const = float(values.pop(0))
    dleak = float(values.pop(0))
    ipakcb = int(values.pop(0))  # ISTCB1 in MODFLOW docs
    istcb2 = int(values.pop(0))

    # Optional unsaturated zone parameters
    isfropt, nstrail, isuzn, nsfrsets = na, na, na, na

    # Handle REACHINPUT option
    if reachinput:
        nstrm = abs(nstrm)  # Per MODFLOW documentation
        if values:
            isfropt = int(values.pop(0))
        if isfropt > 1 and values:
            nstrail = int(values.pop(0)) if values else na
            isuzn = int(values.pop(0)) if values else na
            nsfrsets = int(values.pop(0)) if values else na

    # Handle nstrm < 0 case (unsaturated flow flag)
    if nstrm < 0 and not reachinput:
        if values:
            isfropt = int(values.pop(0))
        if isfropt > 1:
            nstrail = int(values.pop(0)) if values else na
            isuzn = int(values.pop(0)) if values else na
            nsfrsets = int(values.pop(0)) if values else na

    # Transient routing parameters
    irtflg, numtim, weight, flwtol = na, na, na, na

    # VERSION-AWARE: MODFLOW-USG handles transient routing differently
    # In MODFLOW-USG, nstrm < 0 does NOT trigger irtflg reading
    # Only the TRANSROUTE option does
    if version == "mfusg":
        if transroute:
            irtflg = int(pop_item(values, int, 0))
            if irtflg > 0 and values:
                numtim = int(values.pop(0))
                weight = float(values.pop(0))
                flwtol = float(values.pop(0))
    else:
        # MODFLOW-2005/NWT: both nstrm < 0 and transroute trigger irtflg
        if nstrm < 0 or transroute:
            irtflg = int(pop_item(values, int, 0))
            if irtflg > 0 and values:
                numtim = int(values.pop(0))
                weight = float(values.pop(0))
                flwtol = float(values.pop(0))

    # Auxiliary variables (MODFLOW-LGR/NWT)
    option = []
    for i in range(1, len(values)):
        if "aux" in values[i - 1].lower():
            option.append(values[i].lower())

    return (
        nstrm, nss, nsfrpar, nparseg, const, dleak, ipakcb, istcb2,
        isfropt, nstrail, isuzn, nsfrsets, irtflg, numtim, weight, flwtol,
        option,
    )


def get_item2_names(nstrm: int, reachinput: bool, isfropt: int,
                    structured: bool = True) -> List[str]:
    """
    Determine column names for Dataset 2 (reach data).

    Parameters
    ----------
    nstrm : int
        Number of stream reaches (negative indicates special options)
    reachinput : bool
        Whether REACHINPUT option is active
    isfropt : int
        SFR options flag
    structured : bool
        True for structured grid (k,i,j), False for unstructured (node)

    Returns
    -------
    list of str
        Column names for reach data
    """
    names = []

    # Grid location columns
    if structured:
        names += ["k", "i", "j"]
    else:
        names += ["node"]

    # Required reach columns
    names += ["iseg", "ireach", "rchlen"]

    # Optional columns based on isfropt and reachinput
    if nstrm < 0 or reachinput:
        if isfropt in [1, 2, 3]:
            names += ["strtop", "slope", "strthick", "strhc1"]
            if isfropt in [2, 3]:
                names += ["thts", "thti", "eps"]
                if isfropt == 3:
                    names += ["uhc"]

    return names


def parse_6a(line: str, option: List[str]) -> Tuple:
    """
    Parse Dataset 6a for SFR2 package (segment data - first line).

    Parameters
    ----------
    line : str
        Line read from SFR package input file
    option : list of str
        List of auxiliary variable names

    Returns
    -------
    tuple
        All variables for Dataset 6a (17 values including xyz)
    """
    values = line_parse(line)

    # Handle auxiliary variables at end of line
    xyz = []
    for s in values:
        if s.lower() in option:
            xyz.append(s.lower())

    na = 0

    # Required segment parameters
    nseg = int(pop_item(values, int, 0))
    icalc = int(pop_item(values, int, 0))
    outseg = int(pop_item(values, int, 0))
    iupseg = int(pop_item(values, int, 0))

    # Conditional parameters
    iprior = na
    nstrpts = na

    if iupseg > 0:
        iprior = int(pop_item(values, int, 0))
    if icalc == 4:
        nstrpts = int(pop_item(values, int, 0))

    # Flow parameters
    flow = pop_item(values, float, 0.0)
    runoff = pop_item(values, float, 0.0)
    etsw = pop_item(values, float, 0.0)
    pptsw = pop_item(values, float, 0.0)

    # Roughness parameters (conditional on icalc)
    roughch = na
    roughbk = na

    if icalc in [1, 2]:
        roughch = pop_item(values, float, 0.0)
    if icalc == 2:
        roughbk = pop_item(values, float, 0.0)

    # Power equation parameters (icalc == 3)
    cdpth, fdpth, awdth, bwdth = na, na, na, na
    if icalc == 3 and len(values) >= 4:
        cdpth = float(values.pop(0))
        fdpth = float(values.pop(0))
        awdth = float(values.pop(0))
        bwdth = float(values.pop(0))

    return (
        nseg, icalc, outseg, iupseg, iprior, nstrpts,
        flow, runoff, etsw, pptsw, roughch, roughbk,
        cdpth, fdpth, awdth, bwdth, xyz,
    )


def parse_6bc(line: str, icalc: int, nstrm: int, isfropt: int,
              reachinput: bool, per: int = 0) -> Tuple:
    """
    Parse Dataset 6b or 6c for SFR2 package (segment properties).

    This function handles the complex conditional logic for reading
    streambed and unsaturated zone properties based on isfropt and icalc.

    Parameters
    ----------
    line : str
        Line read from SFR package input file
    icalc : int
        Stream depth calculation method
    nstrm : int
        Number of stream reaches
    isfropt : int
        SFR options flag
    reachinput : bool
        Whether REACHINPUT option is active
    per : int
        Stress period index (0-based)

    Returns
    -------
    tuple
        9 values: hcond, thickm, elevupdn, width, depth, thts, thti, eps, uhc
    """
    # Count numeric values in the line
    values_list = line_parse(line)
    nvalues = sum(1 for s in values_list if isnumeric(s))

    # Parse to list
    values = get_dataset(line, [0.0] * nvalues)

    # Initialize output variables
    hcond, thickm, elevupdn, width, depth = 0.0, 0.0, 0.0, 0.0, 0.0
    thts, thti, eps, uhc = 0.0, 0.0, 0.0, 0.0

    # Complex conditional parsing based on isfropt and icalc
    # See MODFLOW Online Guide for Dataset 6b/6c description

    if isfropt in [0, 4, 5] and icalc <= 0:
        # All streambed properties specified
        if values:
            hcond = values.pop(0)
        if values:
            thickm = values.pop(0)
        if values:
            elevupdn = values.pop(0)
        if values:
            width = values.pop(0)
        if values:
            depth = values.pop(0)

    elif isfropt in [0, 4, 5] and icalc == 1:
        if values:
            hcond = values.pop(0)
        # Skip some parameters for later stress periods with isfropt 4 or 5
        if not (isfropt in [4, 5] and per > 0):
            if values:
                thickm = values.pop(0)
            if values:
                elevupdn = values.pop(0)
            if values:
                width = values.pop(0)
            # Unsaturated zone parameters
            thts = pop_item(values, float, 0.0) if values else 0.0
            thti = pop_item(values, float, 0.0) if values else 0.0
            eps = pop_item(values, float, 0.0) if values else 0.0
        if isfropt == 5 and per == 0 and values:
            uhc = values.pop(0)

    elif isfropt in [0, 4, 5] and icalc >= 2:
        if values:
            hcond = values.pop(0)
        if not (isfropt in [4, 5] and per > 0 and icalc == 2):
            if values:
                thickm = values.pop(0)
            if values:
                elevupdn = values.pop(0)
            if isfropt in [4, 5] and per == 0:
                thts = pop_item(values, float, 0.0) if values else 0.0
                thti = pop_item(values, float, 0.0) if values else 0.0
                eps = pop_item(values, float, 0.0) if values else 0.0
                if isfropt == 5:
                    uhc = pop_item(values, float, 0.0) if values else 0.0

    elif isfropt == 1 and icalc <= 1:
        if values:
            width = values.pop(0)
        if icalc <= 0 and values:
            depth = values.pop(0)

    elif isfropt in [2, 3]:
        if icalc <= 0:
            if values:
                width = values.pop(0)
            if values:
                depth = values.pop(0)
        elif icalc == 1:
            if per == 0 and values:
                width = values.pop(0)

    return hcond, thickm, elevupdn, width, depth, thts, thti, eps, uhc


def validate_sfr_options(version: str, nstrm: int, transroute: bool,
                         irtflg: int) -> List[str]:
    """
    Validate SFR options for model version compatibility.

    Parameters
    ----------
    version : str
        Model version ('mf2005', 'mfnwt', 'mfusg')
    nstrm : int
        Number of stream reaches
    transroute : bool
        Whether TRANSROUTE option is active
    irtflg : int
        Transient routing flag

    Returns
    -------
    list of str
        List of warning messages (empty if no issues)
    """
    warnings_list = []

    if version == "mfusg":
        if nstrm < 0 and not transroute and irtflg > 0:
            warnings_list.append(
                "MODFLOW-USG: nstrm < 0 does not enable transient routing. "
                "Use TRANSROUTE option in the options block instead."
            )

    return warnings_list


def markitzero(recarray: np.recarray, inds: List[str]) -> None:
    """
    Subtract 1 from specified columns to convert from 1-based to 0-based indexing.

    Parameters
    ----------
    recarray : np.recarray
        Record array to modify in place
    inds : list of str
        Column names to convert (e.g., ['k', 'i', 'j'] or ['node'])
    """
    lnames = [n.lower() for n in recarray.dtype.names]
    for idx in inds:
        if idx.lower() in lnames:
            recarray[idx] -= 1
