#!/usr/bin/env python3

import pynmrstar
import logging


def remove_software_specific_data(nef_file):
    nef_tags = ['nef_nmr_meta_data', 'nef_molecular_system', 'nef_chemical_shift_list', 'nef_distance_restraint_list',
                'nef_dihedral_restraint_list', 'nef_rdc_restraint_list', 'nef_nmr_spectrum', 'nef_peak_restraint_links',
                'nef_related_entries', 'nef_program_script', 'nef_run_history', 'nef_sequence', 'nef_covalent_links',
                'nef_chemical_shift', 'nef_distance_restraint', 'nef_dihedral_restraint', 'nef_rdc_restraint',
                'nef_spectrum_dimension', 'nef_spectrum_dimension_transfer', 'nef_peak', 'nef_peak_restraint_link',
                'audit', '_nef_program_script.program_name', '_nef_program_script.script_name',
                '_nef_program_script.script',
                '_nef_run_history.run_number', '_nef_run_history.program_name', '_nef_run_history.program_version',
                '_nef_run_history.script_name', '_nef_run_history.script', '_nef_nmr_meta_data.sf_category',
                '_nef_nmr_meta_data.sf_framecode', '_nef_nmr_meta_data.format_name',
                '_nef_nmr_meta_data.format_version',
                '_nef_nmr_meta_data.program_name', '_nef_nmr_meta_data.program_version',
                '_nef_nmr_meta_data.creation_date',
                '_nef_nmr_meta_data.uuid', '_nef_nmr_meta_data.coordinate_file_name',
                '_nef_related_entries.database_name',
                '_nef_related_entries.database_accession_code', '_nef_molecular_system.sf_category',
                '_nef_molecular_system.sf_framecode', '_nef_sequence.index', '_nef_sequence.chain_code',
                '_nef_sequence.sequence_code', '_nef_sequence.residue_name', '_nef_sequence.linking',
                '_nef_sequence.residue_variant', '_nef_sequence.cis_peptide', '_nef_covalent_links.chain_code_1',
                '_nef_covalent_links.sequence_code_1', '_nef_covalent_links.residue_name_1',
                '_nef_covalent_links.atom_name_1', '_nef_covalent_links.chain_code_2',
                '_nef_covalent_links.sequence_code_2',
                '_nef_covalent_links.residue_name_2', '_nef_covalent_links.atom_name_2',
                '_nef_chemical_shift_list.sf_category', '_nef_chemical_shift_list.sf_framecode',
                '_nef_chemical_shift_list.atom_chemical_shift_units', '_nef_chemical_shift.chain_code',
                '_nef_chemical_shift.sequence_code', '_nef_chemical_shift.residue_name',
                '_nef_chemical_shift.atom_name', '_nef_chemical_shift.element', '_nef_chemical_shift.isotope_number',
                '_nef_chemical_shift.value', '_nef_chemical_shift.value_uncertainty',
                '_nef_distance_restraint_list.sf_category', '_nef_distance_restraint_list.sf_framecode',
                '_nef_distance_restraint_list.potential_type', '_nef_distance_restraint_list.restraint_origin',
                '_nef_distance_restraint.index', '_nef_distance_restraint.restraint_id',
                '_nef_distance_restraint.restraint_combination_id', '_nef_distance_restraint.chain_code_1',
                '_nef_distance_restraint.sequence_code_1', '_nef_distance_restraint.residue_name_1',
                '_nef_distance_restraint.atom_name_1', '_nef_distance_restraint.chain_code_2',
                '_nef_distance_restraint.sequence_code_2', '_nef_distance_restraint.residue_name_2',
                '_nef_distance_restraint.atom_name_2', '_nef_distance_restraint.weight',
                '_nef_distance_restraint.target_value', '_nef_distance_restraint.target_value',
                '_nef_distance_restraint.target_value_uncertainty', '_nef_distance_restraint.lower_linear_limit',
                '_nef_distance_restraint.lower_limit', '_nef_distance_restraint.upper_limit',
                '_nef_distance_restraint.upper_linear_limit', '_nef_dihedral_restraint_list.sf_category',
                '_nef_dihedral_restraint_list.sf_framecode', '_nef_dihedral_restraint_list.potential_type',
                '_nef_dihedral_restraint_list.restraint_origin', '_nef_dihedral_restraint.index',
                '_nef_dihedral_restraint.restraint_id', '_nef_dihedral_restraint.restraint_combination_id',
                '_nef_dihedral_restraint.chain_code_1', '_nef_dihedral_restraint.sequence_code_1',
                '_nef_dihedral_restraint.residue_name_1', '_nef_dihedral_restraint.atom_name_1',
                '_nef_dihedral_restraint.chain_code_2', '_nef_dihedral_restraint.sequence_code_2',
                '_nef_dihedral_restraint.residue_name_2', '_nef_dihedral_restraint.atom_name_2',
                '_nef_dihedral_restraint.chain_code_3', '_nef_dihedral_restraint.sequence_code_3',
                '_nef_dihedral_restraint.residue_name_3', '_nef_dihedral_restraint.atom_name_3',
                '_nef_dihedral_restraint.chain_code_4', '_nef_dihedral_restraint.sequence_code_4',
                '_nef_dihedral_restraint.residue_name_4', '_nef_dihedral_restraint.atom_name_4',
                '_nef_dihedral_restraint.weight', '_nef_dihedral_restraint.target_value',
                '_nef_dihedral_restraint.target_value_uncertainty', '_nef_dihedral_restraint.lower_linear_limit',
                '_nef_dihedral_restraint.lower_limit', '_nef_dihedral_restraint.upper_limit',
                '_nef_dihedral_restraint.upper_linear_limit', '_nef_dihedral_restraint.name',
                '_nef_rdc_restraint_list.sf_category', '_nef_rdc_restraint_list.sf_framecode',
                '_nef_rdc_restraint_list.potential_type', '_nef_rdc_restraint_list.restraint_origin',
                '_nef_rdc_restraint_list.tensor_magnitude', '_nef_rdc_restraint_list.tensor_rhombicity',
                '_nef_rdc_restraint_list.tensor_chain_code', '_nef_rdc_restraint_list.tensor_sequence_code',
                '_nef_rdc_restraint_list.tensor_residue_name', '_nef_rdc_restraint.index',
                '_nef_rdc_restraint.restraint_id', '_nef_rdc_restraint.restraint_combination_id',
                '_nef_rdc_restraint.chain_code_1', '_nef_rdc_restraint.sequence_code_1',
                '_nef_rdc_restraint.residue_name_1', '_nef_rdc_restraint.atom_name_1',
                '_nef_rdc_restraint.chain_code_2', '_nef_rdc_restraint.sequence_code_2',
                '_nef_rdc_restraint.residue_name_2', '_nef_rdc_restraint.atom_name_2',
                '_nef_rdc_restraint.weight', '_nef_rdc_restraint.target_value',
                '_nef_rdc_restraint.target_value', '_nef_rdc_restraint.target_value_uncertainty',
                '_nef_rdc_restraint.target_value_uncertainty', '_nef_rdc_restraint.lower_linear_limit',
                '_nef_rdc_restraint.lower_limit', '_nef_rdc_restraint.upper_limit',
                '_nef_rdc_restraint.upper_linear_limit', '_nef_rdc_restraint.scale',
                '_nef_rdc_restraint.distance_dependent', '_nef_nmr_spectrum.sf_category',
                '_nef_nmr_spectrum.sf_framecode', '_nef_nmr_spectrum.num_dimensions',
                '_nef_nmr_spectrum.chemical_shift_list', '_nef_nmr_spectrum.experiment_classification',
                '_nef_nmr_spectrum.experiment_type', '_nef_spectrum_dimension.dimension_id',
                '_nef_spectrum_dimension.axis_unit', '_nef_spectrum_dimension.axis_code',
                '_nef_spectrum_dimension.spectrometer_frequency', '_nef_spectrum_dimension.spectral_width',
                '_nef_spectrum_dimension.value_first_point', '_nef_spectrum_dimension.folding',
                '_nef_spectrum_dimension.absolute_peak_positions', '_nef_spectrum_dimension.is_acquisition',
                '_nef_spectrum_dimension_transfer.dimension_1', '_nef_spectrum_dimension_transfer.dimension_2',
                '_nef_spectrum_dimension_transfer.transfer_type', '_nef_spectrum_dimension_transfer.is_indirect',
                '_nef_peak.index', '_nef_peak.peak_id', '_nef_peak.volume',
                '_nef_peak.volume_uncertainty', '_nef_peak.height', '_nef_peak.height_uncertainty',
                '_nef_peak.position_1', '_nef_peak.position_uncertainty_1', '_nef_peak.position_2',
                '_nef_peak.position_uncertainty_2', '_nef_peak.position_3', '_nef_peak.position_uncertainty_3',
                '_nef_peak.position_4', '_nef_peak.position_uncertainty_4', '_nef_peak.position_5',
                '_nef_peak.position_uncertainty_5', '_nef_peak.position_6', '_nef_peak.position_uncertainty_6',
                '_nef_peak.position_7', '_nef_peak.position_uncertainty_7', '_nef_peak.position_8',
                '_nef_peak.position_uncertainty_8', '_nef_peak.position_9', '_nef_peak.position_uncertainty_9',
                '_nef_peak.position_10', '_nef_peak.position_uncertainty_10', '_nef_peak.position_11',
                '_nef_peak.position_uncertainty_11', '_nef_peak.position_12', '_nef_peak.position_uncertainty_12',
                '_nef_peak.position_13', '_nef_peak.position_uncertainty_13', '_nef_peak.position_14',
                '_nef_peak.position_uncertainty_14', '_nef_peak.position_15', '_nef_peak.position_uncertainty_15',
                '_nef_peak.chain_code_1', '_nef_peak.sequence_code_1', '_nef_peak.residue_name_1',
                '_nef_peak.atom_name_1', '_nef_peak.chain_code_2', '_nef_peak.sequence_code_2',
                '_nef_peak.residue_name_2',
                '_nef_peak.atom_name_2', '_nef_peak.chain_code_3', '_nef_peak.sequence_code_3',
                '_nef_peak.residue_name_3',
                '_nef_peak.atom_name_3', '_nef_peak.chain_code_4', '_nef_peak.sequence_code_4',
                '_nef_peak.residue_name_4',
                '_nef_peak.atom_name_4', '_nef_peak.chain_code_5', '_nef_peak.sequence_code_5',
                '_nef_peak.residue_name_5',
                '_nef_peak.atom_name_5', '_nef_peak.chain_code_6', '_nef_peak.sequence_code_6',
                '_nef_peak.residue_name_6',
                '_nef_peak.atom_name_6', '_nef_peak.chain_code_7', '_nef_peak.sequence_code_7',
                '_nef_peak.residue_name_7',
                '_nef_peak.atom_name_7', '_nef_peak.chain_code_8', '_nef_peak.sequence_code_8',
                '_nef_peak.residue_name_8',
                '_nef_peak.atom_name_8', '_nef_peak.chain_code_9', '_nef_peak.sequence_code_9',
                '_nef_peak.residue_name_9',
                '_nef_peak.atom_name_9', '_nef_peak.chain_code_10', '_nef_peak.sequence_code_10',
                '_nef_peak.residue_name_10',
                '_nef_peak.atom_name_10', '_nef_peak.chain_code_11', '_nef_peak.sequence_code_11',
                '_nef_peak.residue_name_11',
                '_nef_peak.atom_name_11', '_nef_peak.chain_code_12', '_nef_peak.sequence_code_12',
                '_nef_peak.residue_name_12',
                '_nef_peak.atom_name_12', '_nef_peak.chain_code_13', '_nef_peak.sequence_code_13',
                '_nef_peak.residue_name_13',
                '_nef_peak.atom_name_13', '_nef_peak.chain_code_14', '_nef_peak.sequence_code_14',
                '_nef_peak.residue_name_14',
                '_nef_peak.atom_name_14', '_nef_peak.chain_code_15', '_nef_peak.sequence_code_15',
                '_nef_peak.residue_name_15',
                '_nef_peak.atom_name_15', '_nef_peak_restraint_links.sf_category',
                '_nef_peak_restraint_links.sf_framecode',
                '_nef_peak_restraint_link.nmr_spectrum_id', '_nef_peak_restraint_link.peak_id',
                '_nef_peak_restraint_link.restraint_list_id', '_nef_peak_restraint_link.restraint_id',
                '_audit.revision_id', '_audit.creation_date', '_audit.creation_method', '_audit.update_record']

    # Suppress the parse notices, but then log everything after
    logger = logging.getLogger('remove_non_nef_data')
    logger.setLevel(logging.ERROR)
    nef_data = pynmrstar.Entry.from_file(nef_file)
    logger.setLevel(logging.DEBUG)

    saveframe_index = 0
    while saveframe_index < len(nef_data):
        saveframe = nef_data[saveframe_index]
        if not saveframe.category.lower().startswith('nef_'):
            nef_data.remove_saveframe(saveframe)
            logger.debug('Deleted entire saveframe: %s', saveframe.category)
            continue

        tags_to_remove = []
        for tag in saveframe.tags:
            if f'_{saveframe.category}.{tag[0]}'.lower() not in nef_tags:
                tags_to_remove.append(tag)
        saveframe.remove_tag(tags_to_remove)
        logger.debug('Deleted saveframe tags: %s', tags_to_remove)

        loop_index = 0
        while loop_index < len(saveframe):
            loop = saveframe[loop_index]
            if not loop.category.lower().startswith('_nef'):
                del (saveframe[loop_index])
                logger.debug('Deleted entire loop: %s', loop.category)
                continue
            else:
                for tag in loop.get_tag_names():
                    if tag.lower() not in nef_tags:
                        logger.debug('Deleted loop tag: %s', tag)
                        loop.remove_tag(tag)
            loop_index += 1
        saveframe_index += 1

    logger.debug('Finished removing saveframes/loops/tags.')
    nef_data.remove_empty_saveframes()
    out_file = 'cleaned.nef'
    with open(out_file, 'w') as wstarfile:
        wstarfile.write(str(nef_data))
    return True


if __name__ == "__main__":
    remove_software_specific_data('gp78CnewHOME15_test.nef')
