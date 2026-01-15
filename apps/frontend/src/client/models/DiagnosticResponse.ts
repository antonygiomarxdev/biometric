/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type DiagnosticResponse = {
    image_decoded: boolean;
    image_shape: (Array<number> | null);
    image_dtype: (string | null);
    image_stats: (Record<string, number> | null);
    enhancement_completed: boolean;
    enhanced_stats: (Record<string, number> | null);
    extraction_completed: boolean;
    candidates_before_filter: number;
    candidates_after_filter: number;
    skeleton_pixels: (number | null);
    skeleton_ratio: (number | null);
    binary_white_ratio: (number | null);
    error?: (string | null);
};

