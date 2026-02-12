"""
Apply SFR2 parsing patches to FloPy at runtime.

This module monkeypatches the FloPy SFR2 module with improved parsing functions
that handle:
- Better whitespace handling (multiple spaces, tabs)
- MODFLOW-USG node-based models
- Version-aware parsing differences
- More robust error handling

Usage:
    import flopy_patches.apply_patches
    # Now FloPy SFR2 module uses improved parsing

    # Or explicitly:
    from flopy_patches.apply_patches import patch_flopy_sfr
    patch_flopy_sfr()
"""

import warnings
import numpy as np

# Import our improved parsing functions
from . import sfr_parsing


_patched = False


def patch_flopy_sfr():
    """
    Apply SFR2 parsing patches to the installed FloPy package.

    This function replaces the following functions in flopy.modflow.mfsfr2:
    - _isnumeric -> sfr_parsing.isnumeric
    - _pop_item -> sfr_parsing.pop_item
    - _get_dataset -> sfr_parsing.get_dataset
    - _get_item2_names -> sfr_parsing.get_item2_names
    - _parse_1c -> sfr_parsing.parse_1c
    - _parse_6a -> sfr_parsing.parse_6a
    - _parse_6bc -> sfr_parsing.parse_6bc
    - _markitzero -> sfr_parsing.markitzero

    It also patches flopy.utils.flopy_io with improved line parsing.
    """
    global _patched

    if _patched:
        return

    try:
        import flopy.modflow.mfsfr2 as mfsfr2
        import flopy.utils.flopy_io as flopy_io
    except ImportError as e:
        warnings.warn(f"Could not import FloPy modules for patching: {e}")
        return

    # Patch the mfsfr2 module with our improved functions
    mfsfr2._isnumeric = sfr_parsing.isnumeric
    mfsfr2._pop_item = sfr_parsing.pop_item
    mfsfr2._get_dataset = sfr_parsing.get_dataset
    mfsfr2._get_item2_names = sfr_parsing.get_item2_names
    mfsfr2._parse_1c = sfr_parsing.parse_1c
    mfsfr2._parse_6a = sfr_parsing.parse_6a
    mfsfr2._parse_6bc = sfr_parsing.parse_6bc
    mfsfr2._markitzero = sfr_parsing.markitzero

    # Patch the flopy_io module with improved line parsing
    flopy_io.line_strip = sfr_parsing.line_strip
    flopy_io.line_parse = sfr_parsing.line_parse
    flopy_io.pop_item = sfr_parsing.pop_item
    flopy_io.get_next_line = sfr_parsing.get_next_line

    # Also patch the ModflowSfr2.load classmethod to pass version parameter
    _patch_sfr2_load(mfsfr2)

    _patched = True
    print("FloPy SFR2 patches applied successfully")


def _patch_sfr2_load(mfsfr2):
    """
    Patch the ModflowSfr2.load method to properly pass version to _parse_1c.

    This ensures MODFLOW-USG models are parsed correctly.
    """
    original_load = mfsfr2.ModflowSfr2.load

    @classmethod
    def patched_load(cls, f, model, nper=None, gwt=False, nsol=1, ext_unit_dict=None):
        """
        Patched load method that ensures version is passed to parsing functions.
        """
        if model.verbose:
            print("loading sfr2 package file with patched parser...")

        # Import utilities
        from flopy.utils.flopy_io import line_parse, get_next_line, line_strip
        from flopy.utils.optionblock import OptionBlock

        tabfiles = False
        tabfiles_dict = {}
        transroute = False
        reachinput = False
        structured = model.structured
        version = model.version  # Capture model version

        if nper is None:
            nper = model.nper
            nper = 1 if nper == 0 else nper

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # Item 0 -- header (skip comments)
        while True:
            line = f.readline()
            if not line.startswith("#"):
                break

        options = None
        if version == "mfnwt" and "options" in line.lower():
            options = OptionBlock.load_options(f, mfsfr2.ModflowSfr2)
        else:
            query = (
                "reachinput", "transroute", "tabfiles",
                "lossfactor", "strhc1kh", "strhc1kv",
            )
            for i in query:
                if i in line.lower():
                    options = OptionBlock(
                        line.lower().strip(), mfsfr2.ModflowSfr2, block=False
                    )
                    break

        if options is not None:
            line = get_next_line(f)
            if "tabfile" in line.lower():
                t = line.strip().split()
                options.tabfiles = True
                options.numtab = int(t[1])
                options.maxval = int(t[2])
                line = f.readline()

            transroute = options.transroute
            reachinput = options.reachinput
            tabfiles = isinstance(options.tabfiles, np.ndarray)
            numtab = options.numtab if tabfiles else 0

        # Item 1c - USE PATCHED PARSER with version parameter
        (
            nstrm, nss, nsfrpar, nparseg, const, dleak, ipakcb, istcb2,
            isfropt, nstrail, isuzn, nsfrsets, irtflg, numtim, weight, flwtol,
            option,
        ) = sfr_parsing.parse_1c(line, reachinput=reachinput,
                                  transroute=transroute, version=version)

        # Validate options
        warnings_list = sfr_parsing.validate_sfr_options(
            version, nstrm, transroute, irtflg
        )
        for w in warnings_list:
            warnings.warn(w)

        # Item 2 - reach data
        names = sfr_parsing.get_item2_names(nstrm, reachinput, isfropt, structured)
        dtypes = [
            d for d in mfsfr2.ModflowSfr2.get_default_reach_dtype(structured=structured).descr
            if d[0] in names
        ]

        lines = []
        for i in range(abs(nstrm)):
            line = f.readline()
            line = line_parse(line)
            ireach = tuple(map(float, line[: len(dtypes)]))
            lines.append(ireach)

        tmp = np.array(lines, dtype=dtypes)
        reach_data = mfsfr2.ModflowSfr2.get_empty_reach_data(
            len(lines), structured=structured
        )

        for n in names:
            reach_data[n] = tmp[n]

        # Zero-based indexing
        inds = ["k", "i", "j"] if structured else ["node"]
        sfr_parsing.markitzero(reach_data, inds)

        # Items 5 and 6 - stress period data
        segment_data = {}
        channel_geometry_data = {}
        channel_flow_data = {}
        dataset_5 = {}
        aux_variables = {}

        for i in range(0, nper):
            dataset_5[i] = sfr_parsing.get_dataset(
                line_strip(get_next_line(f)), [-1, 0, 0, 0]
            )
            itmp = dataset_5[i][0]

            if itmp > 0:
                current = mfsfr2.ModflowSfr2.get_empty_segment_data(
                    nsegments=itmp, aux_names=option
                )
                current_aux = {}
                current_6d = {}
                current_6e = {}

                for j in range(itmp):
                    dataset_6a = sfr_parsing.parse_6a(get_next_line(f), option)
                    current_aux[j] = dataset_6a[-1]
                    dataset_6a = dataset_6a[:-1]
                    icalc = dataset_6a[1]
                    temp_nseg = dataset_6a[0]

                    dataset_6b, dataset_6c = (0,) * 9, (0,) * 9

                    if not (isfropt in [2, 3] and icalc == 1 and i > 1) and \
                       not (isfropt in [1, 2, 3] and icalc >= 2):
                        dataset_6b = sfr_parsing.parse_6bc(
                            get_next_line(f), icalc, nstrm, isfropt, reachinput, per=i
                        )
                        dataset_6c = sfr_parsing.parse_6bc(
                            get_next_line(f), icalc, nstrm, isfropt, reachinput, per=i
                        )

                    current[j] = dataset_6a + dataset_6b + dataset_6c

                    if icalc == 2:
                        if i == 0 or (nstrm > 0 and not reachinput) or isfropt <= 1:
                            dataset_6d = []
                            for _ in range(2):
                                dataset_6d.append(
                                    sfr_parsing.get_dataset(get_next_line(f), [0.0] * 8)
                                )
                            current_6d[temp_nseg] = dataset_6d

                    if icalc == 4:
                        nstrpts = dataset_6a[5]
                        dataset_6e = []
                        for _ in range(3):
                            dataset_6e.append(
                                sfr_parsing.get_dataset(get_next_line(f), [0.0] * nstrpts)
                            )
                        current_6e[temp_nseg] = dataset_6e

                segment_data[i] = current
                aux_variables[j + 1] = current_aux

                if len(current_6d) > 0:
                    channel_geometry_data[i] = current_6d
                if len(current_6e) > 0:
                    channel_flow_data[i] = current_6e

            if tabfiles and i == 0:
                for j in range(numtab):
                    segnum, numval, iunit = map(
                        int, f.readline().strip().split()[:3]
                    )
                    tabfiles_dict[segnum] = {"numval": numval, "inuit": iunit}

        if openfile:
            f.close()

        # Determine unit number
        import os
        unitnumber = None
        filenames = [None, None, None]
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == mfsfr2.ModflowSfr2._ftype():
                    unitnumber = key
                    filenames[0] = os.path.basename(value.filename)

                if ipakcb > 0:
                    if key == ipakcb:
                        filenames[1] = os.path.basename(value.filename)
                        model.add_pop_key_list(key)

                if abs(istcb2) > 0:
                    if key == abs(istcb2):
                        filenames[2] = os.path.basename(value.filename)
                        model.add_pop_key_list(key)

        return cls(
            model,
            nstrm=nstrm,
            nss=nss,
            nsfrpar=nsfrpar,
            nparseg=nparseg,
            const=const,
            dleak=dleak,
            ipakcb=ipakcb,
            istcb2=istcb2,
            isfropt=isfropt,
            nstrail=nstrail,
            isuzn=isuzn,
            nsfrsets=nsfrsets,
            irtflg=irtflg,
            numtim=numtim,
            weight=weight,
            flwtol=flwtol,
            reach_data=reach_data,
            segment_data=segment_data,
            dataset_5=dataset_5,
            channel_geometry_data=channel_geometry_data,
            channel_flow_data=channel_flow_data,
            reachinput=reachinput,
            transroute=transroute,
            tabfiles=tabfiles,
            tabfiles_dict=tabfiles_dict,
            unit_number=unitnumber,
            filenames=filenames,
            options=options,
        )

    # Apply the patched load method
    mfsfr2.ModflowSfr2.load = patched_load


# Auto-apply patches when module is imported
patch_flopy_sfr()
