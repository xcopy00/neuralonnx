// Modified OnInit function to force CPU execution
void OnInit() {
    // Any existing initialization code
    // ...
    #ifdef GPU
        // Force CPU execution
        ExecuteOnCPU();
    #endif
}