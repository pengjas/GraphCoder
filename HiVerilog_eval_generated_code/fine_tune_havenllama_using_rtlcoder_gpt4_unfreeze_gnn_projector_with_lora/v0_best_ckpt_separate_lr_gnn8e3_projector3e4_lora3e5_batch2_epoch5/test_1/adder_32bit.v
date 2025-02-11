module adder_32bit(A,B,S,C32);

    input [32:1] A,B;
    output [32:1] S;
    output C32;

    wire [16:1] S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16;
    wire C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16;

    CLA CLA1(A[16:1],B[16:1],S1,C1);
    CLA CLA2(A[32:17],B[32:17],S2,C2);
    CLA CLA3(A[32:18],B[32:18],S3,C3);
    CLA CLA4(A[32:19],B[32:19],S4,C4);
    CLA CLA5(A[32:20],B[32:20],S5,C5);
    CLA CLA6(A[32:21],B[32:21],S6,C6);
    CLA CLA7(A[32:22],B[32:22],S7,C7);
    CLA CLA8(A[32:23],B[32:23],S8,C8);
    CLA CLA9(A[32:24],B[32:24],S9,C9);
    CLA CLA10(A[32:25],B[32:25],S10,C10);
    CLA CLA11(A[32:26],B[32:26],S11,C11);
    CLA CLA12(A[32:27],B[32:27],S12,C12);
    CLA CLA13(A[32:28],B[32:28],S13,C13);
    CLA CLA14(A[32:29],B[32:29],S14,C14);
    CLA CLA15(A[32:30],B[32:30],S15,C15);
    CLA CLA16(A[32:31],B[32:31],S16,C16);

    assign S = {S16,S15,S14,S13,S12,S11,S10,S9,S8,S7,S6,S5,S4,S3,S2,S1};
    assign C32 = C16;

endmodule

module CLA(A,B,S,C);
    input [16:1] A,B;
    output [16:1] S;
    output C;

    wire [16:1] S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16;
    wire C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16;

    CLA CLA1(A[8:1],B[8:1],S1,C1);
    CLA CLA2(A[16:9],B[16:9],S2,C2);
    CLA CLA3(A[16:10],B[16:10],S3,C3);
    CLA CLA4(A[16:11],B[16:11
endmodule