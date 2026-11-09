export interface Recording {
    uuid: string;
    voiced?: Blob;
    unvoiced?: Blob;
    text: string;
    name?: string;
    timestamp: string;
}
