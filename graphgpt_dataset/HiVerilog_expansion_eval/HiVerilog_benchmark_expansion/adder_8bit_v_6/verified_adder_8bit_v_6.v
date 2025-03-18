module subtractor_8bit(
    input [7:0] a, b,
    input bin,
    output [7:0] diff,
    output bout);

    wire [8:0] btemp;

    full_subtractor FS0 (.a(a[0]), .b(b[0]), .bin(bin), .diff(diff[0]), .bout(btemp[0]));
    full_subtractor FS1 (.a(a[1]), .b(b[1]), .bin(btemp[0]), .diff(diff[1]), .bout(btemp[1]));
    full_subtractor FS2 (.a(a[2]), .b(b[2]), .bin(btemp[1]), .diff(diff[2]), .bout(btemp[2]));
    full_subtractor FS3 (.a(a[3]), .b(b[3]), .bin(btemp[2]), .diff(diff[3]), .bout(btemp[3]));
    full_subtractor FS4 (.a(a[4]), .b(b[4]), .bin(btemp[3]), .diff(diff[4]), .bout(btemp[4]));
    full_subtractor FS5 (.a(a[5]), .b(b[5]), .bin(btemp[4]), .diff(diff[5]), .bout(btemp[5]));
    full_subtractor FS6 (.a(a[6]), .b(b[6]), .bin(btemp[5]), .diff(diff[6]), .bout(btemp[6]));
    full_subtractor FS7 (.a(a[7]), .b(b[7]), .bin(btemp[6]), .diff(diff[7]), .bout(btemp[7]));

    assign bout = btemp[7];
endmodule

module full_subtractor (input a, b, bin, output diff, bout);
    assign {bout, diff} = a - b - bin;
endmodule
