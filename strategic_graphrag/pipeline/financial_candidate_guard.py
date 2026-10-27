"""Conservative context checks before consolidating table candidates.

This quarantines unresolved context, without repairing values from references.
The original candidates must remain in the build's quarantine audit.
"""
import re


def rejection_reason(triple):
    if triple.get('relation') != 'REPORTS_METRIC':
        return None
    if triple.get('financial_basis')=='UNKNOWN':
        return 'financial_basis_not_established_by_source_context'
    if triple.get('column_semantics')=='SEGMENT_COLUMNS_UNRESOLVED':
        return 'segment_columns_cannot_substitute_for_period_columns'
    # The legacy metric conflates basic and diluted EPS. A verbatim row alone
    # is insufficient when that basis is absent from the canonical identity.
    # Preserve candidates in quarantine until the schema/query contract can
    # represent the basis; never silently answer a basic request with diluted.
    if triple.get('target') == 'EARNINGS_PER_SHARE':
        return 'earnings_per_share_basis_not_in_fact_identity'
    title = str(triple.get('table_name') or '').casefold()
    if re.search(r'deferred tax (?:assets|liabilities)',title):
        return 'deferred_tax_component_is_not_underlying_company_balance'
    unit = str(triple.get('metric_unit') or triple.get('unit') or '').casefold()
    if re.search(r'\ballocation\b|purchase.price|acquisition.date', title):
        return 'acquisition_allocation_is_not_company_year_end_balance'
    if 'graphics' in title and ('networking' in title or 'all other' in title):
        return 'segment_columns_not_resolved_to_year_columns'
    if ('percentage of revenue' in title or 'percent of revenue' in title) and unit not in {'percent','%','percentage points'}:
        return 'percentage_table_conflicts_with_amount_unit'
    return None
