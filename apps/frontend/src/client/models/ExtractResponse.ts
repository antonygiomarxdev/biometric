/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MinutiaPoint } from './MinutiaPoint';
export type ExtractResponse = {
    minutiae_count: number;
    terminations: number;
    bifurcations: number;
    minutiae: Array<MinutiaPoint>;
    processing_time_ms?: (number | null);
    processed_image?: (string | null);
    image_shape?: (Array<number> | null);
    image_dtype?: (string | null);
    minutiae_initial_count?: (number | null);
};

