module magnitude_comparator (A, B, gt, lt, eq);
    input [3:0] A;
    input [3:0] B;
    output gt, lt, eq;
    wire gt0, gt1, gt2, gt3;
    wire lt0, lt1, lt2, lt3;
    wire eq0, eq1, eq2, eq3;

    // Bit-by-bit comparison
    bit_compare bc0 (A[3], B[3], gt3, lt3, eq3);
    bit_compare bc1 (A[2], B[2], gt2, lt2, eq2);
    bit_compare bc2 (A[1], B[1], gt1, lt1, eq1);
    bit_compare bc3 (A[0], B[0], gt0, lt0, eq0);

    // Output determination
    assign gt = gt3 | (eq3 & gt2) | (eq3 & eq2 & gt1) | (eq3 & eq2 & eq1 & gt0);
    assign lt = lt3 | (eq3 & lt2) | (eq3 & eq2 & lt1) | (eq3 & eq2 & eq1 & lt0);
    assign eq = eq3 & eq2 & eq1 & eq0;

endmodule

module bit_compare(a, b, gt, lt, eq);
    input a, b;
    output gt, lt, eq;

    assign gt = a & ~b;
    assign lt = ~a & b;
    assign eq = a ~^ b; // XOR for equality check
endmodule
